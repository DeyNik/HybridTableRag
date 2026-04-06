"""
reasoning/intent.py
===================
Query planning via a single LLM call that returns a structured QueryPlan.

Design principles
-----------------
1. ONE LLM call produces the full plan — no separate classify() + classify_python_mode().
2. The prompt explains the system's capabilities in neutral terms, not rigid keywords.
   The LLM infers intent from semantics, not pattern matching.
3. QueryPlan carries all decisions the orchestrator needs:
     - which tables are relevant
     - whether aggregation is needed
     - whether visualization would help
     - whether semantic/vector search is needed
     - conversational flag
4. Keyword fallback only fires if the LLM call fails — it's a safety net, not the engine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


# QueryPlan

@dataclass
class QueryPlan:
    """
    Full execution plan produced by one LLM call.
    The orchestrator reads this and decides what to run — nothing is hardcoded in the orchestrator.
    """
    # Core routing
    needs_sql:             bool = True    # almost always True unless pure conversational
    needs_python:          bool = False   # True when visualization or complex stats add value
    needs_semantic:        bool = False   # True when the question asks for meaning-based search
    is_conversational:     bool = False   # True when asking about the conversation itself

    # SQL hints (help sql.py build a better prompt)
    relevant_tables:       List[str] = field(default_factory=list)   # tables the LLM thinks are needed
    needs_aggregation:     bool = False   # GROUP BY / COUNT / SUM / AVG involved
    needs_join:            bool = False   # multiple tables need to be joined
    needs_ranking:         bool = False   # ORDER BY + LIMIT / RANK / ROW_NUMBER

    # Python / visualization hints
    python_mode:           str = "table"  # "chart" | "stats" | "both" | "table"
    visualization_reason:  str = ""       # why a chart would help (used in python prompt)

    # Semantic search hints
    semantic_column_hint:  str = ""       # which text column to search (empty = auto-detect)
    semantic_query:        str = ""       # rephrased query optimized for embedding search

    # Raw LLM reasoning (useful for debug mode)
    reasoning:             str = ""


# Prompt 
# The prompt describes capabilities neutrally — the LLM decides applicability.
# No keyword lists, no rigid rules. The LLM has full context about the system.

_PLAN_PROMPT = """\
You are a query planner for a data analytics system that works with tabular data \
(CSV/Excel files loaded into a relational database).

The system can:
1. Execute SQL queries against structured tables (filters, counts, aggregates, joins, ranking)
2. Run Python/pandas analysis and generate Plotly visualizations
3. Perform semantic similarity search over text columns using vector embeddings
4. Answer conversational questions about the chat history

Available tables and their columns:
{schema_summary}

User question:
{query}

Decide the best execution plan. Think about:
- Which tables contain the data needed to answer this question
- Whether the answer requires grouping/aggregating data
- Whether the answer would be significantly clearer with a chart or graph
  (e.g. comparisons, trends, distributions, proportions — not just a count)
- Whether the user is searching for semantically similar content in text columns
  (e.g. "find tickets about login issues" vs "count tickets by status")
- Whether this is a follow-up question about the conversation itself

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "needs_sql": true,
  "needs_python": false,
  "needs_semantic": false,
  "is_conversational": false,
  "relevant_tables": ["table1"],
  "needs_aggregation": false,
  "needs_join": false,
  "needs_ranking": false,
  "python_mode": "table",
  "visualization_reason": "",
  "semantic_column_hint": "",
  "semantic_query": "",
  "reasoning": "one sentence explaining the plan"
}}
"""


# Classifier

class IntentClassifier:
    """
    Produces a QueryPlan from a single LLM call.
    Falls back to heuristics if the LLM call fails or returns unparseable output.
    """

    def __init__(self, llm):
        self.llm = llm

    def classify(
        self,
        user_query: str,
        schema_summary: str = "",
        available_tables: Optional[List[str]] = None,
    ) -> QueryPlan:
        """
        Classify user_query into a QueryPlan.

        schema_summary: compact text description of available tables/columns
                        (from storage/schema.py → format_schema_for_prompt)
        available_tables: list of table names (used for fallback + validation)
        """
        available_tables = available_tables or []

        try:
            prompt = _PLAN_PROMPT.format(
                schema_summary=schema_summary or "(schema not provided)",
                query=user_query,
            )
            raw = self.llm.generate(prompt).strip()

            # Strip markdown fences if LLM added them despite instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw).strip()

            parsed = json.loads(raw)
            plan = self._parse_plan(parsed, available_tables)
            return plan

        except Exception as e:
            # LLM call or JSON parse failed — use keyword heuristics as safety net
            return self._fallback_plan(user_query, available_tables, error=str(e))

    def _parse_plan(self, parsed: dict, available_tables: List[str]) -> QueryPlan:
        """Convert LLM JSON dict to QueryPlan, with validation."""
        plan = QueryPlan(
            needs_sql=bool(parsed.get("needs_sql", True)),
            needs_python=bool(parsed.get("needs_python", False)),
            needs_semantic=bool(parsed.get("needs_semantic", False)),
            is_conversational=bool(parsed.get("is_conversational", False)),
            needs_aggregation=bool(parsed.get("needs_aggregation", False)),
            needs_join=bool(parsed.get("needs_join", False)),
            needs_ranking=bool(parsed.get("needs_ranking", False)),
            python_mode=str(parsed.get("python_mode", "table")),
            visualization_reason=str(parsed.get("visualization_reason", "")),
            semantic_column_hint=str(parsed.get("semantic_column_hint", "")),
            semantic_query=str(parsed.get("semantic_query", "")),
            reasoning=str(parsed.get("reasoning", "")),
        )

        # Validate and filter relevant_tables to only known tables
        raw_tables = parsed.get("relevant_tables", [])
        if available_tables and raw_tables:
            plan.relevant_tables = [t for t in raw_tables if t in available_tables]
        else:
            plan.relevant_tables = raw_tables

        # If no tables selected but we have some, default to all
        if not plan.relevant_tables and available_tables:
            plan.relevant_tables = available_tables

        # If conversational, turn off SQL
        if plan.is_conversational:
            plan.needs_sql = False
            plan.needs_python = False
            plan.needs_semantic = False

        # Validate python_mode
        if plan.python_mode not in {"chart", "stats", "both", "table"}:
            plan.python_mode = "table"

        return plan

    def _fallback_plan(
        self,
        query: str,
        available_tables: List[str],
        error: str = "",
    ) -> QueryPlan:
        """
        Keyword-based fallback. Only runs if LLM fails.
        Deliberately simple — catches obvious cases, defaults to SQL.
        """
        q = query.lower()

        plan = QueryPlan(
            relevant_tables=available_tables,
            reasoning=f"Fallback plan (LLM error: {error[:60]})" if error else "Fallback plan",
        )

        # Conversational
        if any(k in q for k in ["what did i ask", "earlier", "previous", "our conversation", "summarise our"]):
            plan.is_conversational = True
            plan.needs_sql = False
            return plan

        # Visualization signals
        viz_signals = ["compare", "versus", "vs ", "trend", "over time", "distribution",
                       "breakdown", "proportion", "percentage", "by month", "by week",
                       "plot", "chart", "graph", "visuali"]
        if any(k in q for k in viz_signals):
            plan.needs_python = True
            plan.python_mode = "both" if any(k in q for k in ["stat", "correlation", "outlier"]) else "chart"

        # Semantic search signals
        semantic_signals = ["find tickets about", "find rows about", "similar to", "like this",
                            "search for", "mentions", "describes", "talks about", "related to"]
        if any(k in q for k in semantic_signals):
            plan.needs_semantic = True
            plan.semantic_query = query

        # Aggregation signals
        agg_signals = ["how many", "count", "total", "sum", "average", "avg", "mean",
                       "maximum", "minimum", "most", "least", "top", "bottom", "rank"]
        if any(k in q for k in agg_signals):
            plan.needs_aggregation = True

        # Join signals — if query mentions multiple table concepts
        if len(available_tables) > 1:
            plan.needs_join = any(
                t.replace("_", " ") in q for t in available_tables[1:]
            )

        return plan
"""
reasoning/intent.py
===================
Intent classification for query routing.
"""

INTENT_PROMPT = """
You are a query router for a data analytics system.

The system has four execution paths:
  sql           → structured data queries (filters, counts, aggregations, ranking)
  python        → charts, plots, statistical analysis, correlations, distributions
  vector        → semantic search: "find similar", "search by meaning"
  conversational → questions about the conversation itself

Respond with ONLY one word:
sql OR python OR vector OR conversational

User question: {query}
"""


PYTHON_MODE_PROMPT = """
Classify what kind of Python analysis is needed.

Options:
  chart  → visualization (plot, graph, chart)
  stats  → statistical analysis (correlation, distribution, outliers)
  both   → both chart + stats
  table  → simple dataframe transformation

Respond with ONLY one word: chart OR stats OR both OR table

User question: {query}
"""


class IntentClassifier:
    VALID_INTENTS = {"sql", "python", "vector", "conversational"}
    DEFAULT = "sql"

    def __init__(self, llm):
        self.llm = llm

    # ─────────────────────────────────────────────────────────────
    # Main intent classification
    # ─────────────────────────────────────────────────────────────

    def classify(self, user_query: str) -> str:
        try:
            prompt = INTENT_PROMPT.format(query=user_query)
            response = self.llm.generate(prompt)

            if not response:
                return self._fallback(user_query)

            text = response.strip().lower()
            text = text.replace(".", "").strip()

            # exact match
            if text in self.VALID_INTENTS:
                return text

            # partial match
            for intent in self.VALID_INTENTS:
                if intent in text:
                    return intent

            return self._fallback(user_query)

        except Exception:
            return self._fallback(user_query)

    # ─────────────────────────────────────────────────────────────
    # Python mode classification
    # ─────────────────────────────────────────────────────────────

    def classify_python_mode(self, user_query: str) -> str:
        try:
            prompt = PYTHON_MODE_PROMPT.format(query=user_query)
            response = self.llm.generate(prompt)

            if not response:
                return self._python_fallback(user_query)

            text = response.strip().lower()
            text = text.replace(".", "").strip()

            if text in {"chart", "stats", "both", "table"}:
                return text

            for mode in ["chart", "stats", "both", "table"]:
                if mode in text:
                    return mode

            return self._python_fallback(user_query)

        except Exception:
            return self._python_fallback(user_query)

    # ─────────────────────────────────────────────────────────────
    # Fallbacks
    # ─────────────────────────────────────────────────────────────

    def _fallback(self, query: str) -> str:
        q = query.lower()

        # conversational (highest priority)
        if any(k in q for k in ["earlier", "before", "previous", "explain", "what did i"]):
            return "conversational"

        # python
        if any(k in q for k in ["plot", "chart", "graph", "visual", "trend"]):
            return "python"

        # vector (more strict to avoid false positives)
        if any(k in q for k in ["find similar", "semantic", "like this", "search by meaning"]):
            return "vector"

        return "sql"

    def _python_fallback(self, query: str) -> str:
        q = query.lower()

        if any(k in q for k in ["chart", "plot", "graph"]):
            return "chart"

        if any(k in q for k in ["correlation", "distribution", "outlier", "stat"]):
            return "stats"

        if "and" in q and any(k in q for k in ["plot", "chart"]):
            return "both"

        return "table"
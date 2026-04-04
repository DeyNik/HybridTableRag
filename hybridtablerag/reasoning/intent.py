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

Respond with ONLY one word: sql OR python OR vector OR conversational
User question: {query}
"""

PYTHON_MODE_PROMPT = """
Classify what kind of Python analysis is needed.
Options: chart, stats, both, table
Respond with ONLY one word.
User question: {query}
"""


class IntentClassifier:
    VALID_INTENTS = {"sql", "python", "vector", "conversational"}
    DEFAULT = "sql"

    def __init__(self, llm):
        self.llm = llm

    def classify(self, user_query: str) -> str:
        try:
            response = self.llm.generate(INTENT_PROMPT.format(query=user_query))
            if not response: return self._fallback(user_query)
            text = response.strip().lower().replace(".", "").strip()
            if text in self.VALID_INTENTS: return text
            for intent in self.VALID_INTENTS:
                if intent in text: return intent
            return self._fallback(user_query)
        except Exception:
            return self._fallback(user_query)

    def classify_python_mode(self, user_query: str) -> str:
        try:
            response = self.llm.generate(PYTHON_MODE_PROMPT.format(query=user_query))
            if not response: return self._python_fallback(user_query)
            text = response.strip().lower().replace(".", "").strip()
            if text in {"chart", "stats", "both", "table"}: return text
            for mode in ["chart", "stats", "both", "table"]:
                if mode in text: return mode
            return self._python_fallback(user_query)
        except Exception:
            return self._python_fallback(user_query)

    def _fallback(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["earlier", "before", "previous", "explain", "what did i", "summarize"]):
            return "conversational"
        if any(k in q for k in ["plot", "chart", "graph", "visual", "trend", "distribution"]):
            return "python"
        if any(k in q for k in ["find similar", "semantic", "like this", "search by meaning", "find rows where"]):
            return "vector"
        return "sql"

    def _python_fallback(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["correlation", "distribution", "outlier", "stat"]):
            return "stats"
        if any(k in q for k in ["chart", "plot", "graph"]):
            return "chart" if "and" not in q else "both"
        return "table"
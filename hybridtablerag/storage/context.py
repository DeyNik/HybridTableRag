"""
storage/context.py
==================
Conversation history stored in DuckDB.
"""

from typing import List, Optional
import pandas as pd


class ContextStore:
    TABLE_NAME = "chat_history"

    def __init__(self, conn):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                session_id   VARCHAR,
                turn         INTEGER,
                timestamp    TIMESTAMP DEFAULT now(),
                user_query   VARCHAR,
                intent       VARCHAR,
                sql_generated VARCHAR,
                python_code  VARCHAR,
                result_summary VARCHAR,
                error        VARCHAR,
                reasoning    VARCHAR
            )
        """)

        self.conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_chat_session
        ON chat_history(session_id, turn);        
        """)

    def save_turn(
        self,
        session_id: str,
        user_query: str,
        intent: str,
        result_summary: str,
        sql_generated: Optional[str] = None,
        python_code: Optional[str] = None,
        error: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> int:

        # --- get next turn ---
        result = self.conn.execute(
            f"""
            SELECT COALESCE(MAX(turn), 0)
            FROM {self.TABLE_NAME}
            WHERE session_id = ?
            """,
            [session_id]
        ).fetchone()

        next_turn = result[0] + 1

        # --- insert ---
        self.conn.execute(
            f"""
            INSERT INTO {self.TABLE_NAME} (
                session_id, turn, user_query, intent,
                sql_generated, python_code, result_summary,
                error, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                session_id,
                next_turn,
                user_query,
                intent,
                sql_generated,
                python_code,
                result_summary,
                error,
                reasoning
            ]
        )

        return next_turn

    def get_history(
        self,
        session_id: str,
        last_n: int = 5,
    ) -> List[dict]:

        df = self.conn.execute(
            f"""
            SELECT *
            FROM {self.TABLE_NAME}
            WHERE session_id = ?
            ORDER BY turn DESC
            LIMIT ?
            """,
            [session_id, last_n]
        ).fetchdf()

        # return oldest first
        df = df.sort_values("turn")

        return df.to_dict("records")

    def build_context_summary(
        self,
        session_id: str,
        last_n: int = 3,
    ) -> str:

        history = self.get_history(session_id, last_n=last_n)

        if not history:
            return ""

        lines = ["Previous conversation:"]

        for h in history:
            summary = h.get("result_summary") or "no result"
            query = h.get("user_query")

            lines.append(
                f"Turn {h['turn']}: User asked '{query}' → {summary}"
            )

        return "\n".join(lines)

    def clear_session(self, session_id: str):
        self.conn.execute(
            f"""
            DELETE FROM {self.TABLE_NAME}
            WHERE session_id = ?
            """,
            [session_id]
        )

    def get_all_sessions(self) -> pd.DataFrame:
        return self.conn.execute(
            f"""
            SELECT
                session_id,
                COUNT(*) as turns,
                MIN(timestamp) as started_at,
                MAX(timestamp) as last_activity
            FROM {self.TABLE_NAME}
            GROUP BY session_id
            ORDER BY last_activity DESC
            """
        ).fetchdf()
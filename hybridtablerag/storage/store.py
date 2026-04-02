"""
storage/store.py
================
DuckDB connection manager and table registration.
"""

import os
import duckdb
from typing import List
import pandas as pd
from hybridtablerag.core.normalizer import NormalizationPlan


class DuckDBStore:
    """
    Central DuckDB connection. One instance per session.
    """

    def __init__(self, db_path: str = "data/hybridtablerag.duckdb"):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        absolute_db_path = os.path.join(project_root, db_path)
        os.makedirs(os.path.dirname(absolute_db_path), exist_ok=True)

        self.db_path = absolute_db_path
        self.conn = duckdb.connect(self.db_path)

    # ─────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────

    def register_normalization_plan(
        self,
        plan: NormalizationPlan,
        bts_log: list,
    ) -> None:
        """
        Register main + bridge tables safely into DuckDB.
        """

        # Register ALL tables (main + bridges)
        for table_name, table_obj in plan.all_tables.items():
            df = table_obj.df

            tmp_name = "_tmp_register"

            self.conn.register(tmp_name, df)
            self.conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp_name}"
            )
            self.conn.unregister(tmp_name)

            bts_log.append(f"Registered table: {table_name} ({len(df)} rows)")

    def register_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        bts_log: list,
    ) -> None:
        """
        Register a single DataFrame (no normalization).
        """

        tmp_name = "_tmp_cleaned"

        self.conn.register(tmp_name, df)
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp_name}"
        )
        self.conn.unregister(tmp_name)

        bts_log.append(f"Registered table: {table_name} ({len(df)} rows)")

    # ─────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────

    def list_tables(self) -> List[str]:
        """
        Return all table names.
        """
        result = self.conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]

    def get_table_schema(self, table_name: str) -> List[dict]:
        """
        Return schema info.
        """
        result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()

        columns = []
        for row in result:
            columns.append({
                "column_name": row[0],
                "data_type": row[1],
                "null": row[2],
                "key": row[3],
                "default": row[4],
                "extra": row[5],
            })

        return columns

    # ─────────────────────────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────────────────────────

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL safely.
        """
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            raise RuntimeError(f"SQL execution failed:\n{sql}\nError: {str(e)}")

    # ─────────────────────────────────────────────────────────────
    # Relationship inference (cross-table)
    # ─────────────────────────────────────────────────────────────

    def infer_relationships(self) -> List[dict]:
        """
        Infer relationships based on column name + uniqueness heuristics.
        """

        relationships = []
        tables = self.list_tables()

        # Collect schema
        schemas = {}
        for t in tables:
            schemas[t] = self.get_table_schema(t)

        for t1 in tables:
            for t2 in tables:
                if t1 == t2:
                    continue

                cols1 = schemas[t1]
                cols2 = schemas[t2]

                for c1 in cols1:
                    for c2 in cols2:

                        if (
                            c1["column_name"] == c2["column_name"]
                            and c1["data_type"] == c2["data_type"]
                        ):
                            col = c1["column_name"]

                            try:
                                # Check uniqueness in t2
                                col_escaped = f'"{col}"'
                                unique_count = self.conn.execute(
                                    f"SELECT COUNT(DISTINCT {col_escaped}) FROM {t2}"
                                ).fetchone()[0]

                                total_count = self.conn.execute(
                                    f"SELECT COUNT(*) FROM {t2}"
                                ).fetchone()[0]

                                if unique_count == total_count:
                                    relationships.append({
                                        "from_table": t1,
                                        "from_column": col,
                                        "to_table": t2,
                                        "to_column": col,
                                        "type": "many_to_one"
                                    })

                            except Exception:
                                continue

        return relationships

    # ─────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
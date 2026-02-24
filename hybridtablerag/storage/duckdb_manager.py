import os
import duckdb
from typing import Dict, List, Any


class DuckDBManager:
    def __init__(self, db_path: str = "data/hybridtablerag.duckdb"):
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            absolute_db_path = os.path.join(project_root, db_path)

            os.makedirs(os.path.dirname(absolute_db_path), exist_ok=True)

            self.db_path = absolute_db_path
            self.conn = duckdb.connect(self.db_path)

            print(f"[DuckDB] Initialized at {self.db_path}")

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Initialization failed: {e}")

    def register_csv(self, file_path: str, table_name: str):
        try:
            absolute_path = os.path.abspath(file_path)

            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"CSV file not found: {absolute_path}")

            print(f"[DuckDB] Registering table: {table_name}")

            query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{absolute_path}', HEADER=TRUE)
            """

            self.conn.execute(query)

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Failed to register CSV: {e}")

    def list_tables(self) -> List[str]:
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            raise RuntimeError(f"[DuckDB] Failed to list tables: {e}")

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()

            schema = []
            for row in result:
                schema.append({
                    "column_name": row[0],
                    "data_type": row[1],
                    "null": row[2],
                    "key": row[3],
                    "default": row[4],
                    "extra": row[5]
                })

            return schema

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Failed to get schema for '{table_name}': {e}")

    def get_database_schema_overview(self) -> Dict[str, Any]:
        try:
            overview = {}
            tables = self.list_tables()

            for table in tables:
                overview[table] = {
                    "columns": self.get_table_schema(table)
                }

            return overview

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Failed to build schema overview: {e}")

    def execute_query(self, query: str):
        try:
            return self.conn.execute(query).fetchdf()
        except Exception as e:
            raise RuntimeError(f"[DuckDB] Query execution failed: {e}")

    def close(self):
        try:
            self.conn.close()
            print("[DuckDB] Connection closed.")
        except Exception:
            pass
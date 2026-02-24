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

    def infer_relationships(self):
        """
        Infer potential relationships between tables
        based on matching column names and data types.
        Returns a structured relationship graph.
        """
        try:
            schema_overview = self.get_database_schema_overview()
            tables = list(schema_overview.keys())

            relationships = {}

            for table_a in tables:
                relationships[table_a] = []

                columns_a = schema_overview[table_a]["columns"]

                for table_b in tables:
                    if table_a == table_b:
                        continue

                    columns_b = schema_overview[table_b]["columns"]

                    for col_a in columns_a:
                        for col_b in columns_b:
                            if (
                                col_a["column_name"] == col_b["column_name"]
                                and col_a["data_type"] == col_b["data_type"]
                            ):
                                relationships[table_a].append({
                                    "from_column": col_a["column_name"],
                                    "to_table": table_b,
                                    "to_column": col_b["column_name"],
                                    "data_type": col_a["data_type"]
                                })

            return relationships

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Relationship inference failed: {e}")
        

    def suggest_joins(self, tables):
        """
        Suggest JOIN conditions between provided tables
        based on inferred relationships.
        """
        try:
            if not tables or len(tables) < 2:
                return []

            relationships = self.infer_relationships()
            joins = []

            for table in tables:
                if table not in relationships:
                    continue

                for rel in relationships[table]:
                    if rel["to_table"] in tables:
                        condition = (
                            f"{table}.{rel['from_column']} = "
                            f"{rel['to_table']}.{rel['to_column']}"
                        )

                        # Avoid duplicate reversed joins
                        reverse_condition = (
                            f"{rel['to_table']}.{rel['to_column']} = "
                            f"{table}.{rel['from_column']}"
                        )

                        if condition not in joins and reverse_condition not in joins:
                            joins.append(condition)

            return joins

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Join suggestion failed: {e}")
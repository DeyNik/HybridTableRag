import os
import duckdb
from typing import Dict, List, Any
import re

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
        

    def profile_tables(self):
        """
        Generate profiling metadata for all tables.
        Includes row count, column stats, and distinct counts.
        """
        try:
            schema_overview = self.get_database_schema_overview()
            profiling_data = {}

            for table, table_info in schema_overview.items():
                profiling_data[table] = {}
                
                # Row count
                row_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

                profiling_data[table]["row_count"] = row_count
                profiling_data[table]["columns"] = {}

                for column in table_info["columns"]:
                    col_name = column["column_name"]
                    col_type = column["data_type"]

                    column_profile = {
                        "data_type": col_type
                    }

                    # Null count
                    null_count = self.conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL"
                    ).fetchone()[0]

                    column_profile["null_count"] = null_count

                    # Numeric stats
                    if any(t in col_type.upper() for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL"]):
                        stats = self.conn.execute(
                            f"SELECT MIN({col_name}), MAX({col_name}) FROM {table}"
                        ).fetchone()

                        column_profile["min"] = stats[0]
                        column_profile["max"] = stats[1]

                    # Date stats
                    elif "DATE" in col_type.upper():
                        stats = self.conn.execute(
                            f"SELECT MIN({col_name}), MAX({col_name}) FROM {table}"
                        ).fetchone()

                        column_profile["min_date"] = stats[0]
                        column_profile["max_date"] = stats[1]

                    # Distinct count (for categorical / general)
                    distinct_count = self.conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                    ).fetchone()[0]

                    column_profile["distinct_count"] = distinct_count

                    profiling_data[table]["columns"][col_name] = column_profile

            return profiling_data

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Profiling failed: {e}")
        

    def _safe_serialize(self, value):
        """
        Ensures all values are JSON serializable.
        Handles dates, datetimes, decimals and unknown types.
        """
        try:
            import datetime
            from decimal import Decimal

            if value is None:
                return None

            if isinstance(value, (datetime.date, datetime.datetime)):
                return value.isoformat()

            if isinstance(value, Decimal):
                return float(value)

            # Handle numpy types if present
            try:
                import numpy as np
                if isinstance(value, (np.integer, np.floating)):
                    return value.item()
            except Exception:
                pass

            return value

        except Exception:
            return str(value)
        
    def build_structured_schema_metadata(self):
        """
        Builds structured, JSON-safe metadata for all tables.
        """

        try:
            tables = self.list_tables()
            metadata = []

            for table in tables:
                row_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

                table_meta = {
                    "table_name": table,
                    "row_count": self._safe_serialize(row_count),
                    "columns": []
                }

                schema = self.get_table_schema(table)

                for col in schema:
                    col_name = col["column_name"]
                    col_type = col["data_type"]

                    column_meta = {
                        "name": col_name,
                        "type": col_type,
                    }

                    # Nullability
                    null_count = self.conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL"
                    ).fetchone()[0]

                    column_meta["nullable"] = null_count > 0

                    # Distinct count
                    distinct_count = self.conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                    ).fetchone()[0]

                    column_meta["distinct_count"] = self._safe_serialize(distinct_count)

                    # Unique detection
                    column_meta["unique"] = distinct_count == row_count

                    # Numeric stats
                    if any(t in col_type.upper() for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL"]):
                        stats = self.conn.execute(
                            f"SELECT MIN({col_name}), MAX({col_name}) FROM {table}"
                        ).fetchone()

                        column_meta["min"] = self._safe_serialize(stats[0])
                        column_meta["max"] = self._safe_serialize(stats[1])

                    # Date stats
                    if "DATE" in col_type.upper():
                        stats = self.conn.execute(
                            f"SELECT MIN({col_name}), MAX({col_name}) FROM {table}"
                        ).fetchone()

                        column_meta["min"] = self._safe_serialize(stats[0])
                        column_meta["max"] = self._safe_serialize(stats[1])

                    # Sample values
                    sample_rows = self.conn.execute(
                        f"""
                        SELECT {col_name}
                        FROM {table}
                        WHERE {col_name} IS NOT NULL
                        LIMIT 3
                        """
                    ).fetchall()

                    column_meta["sample_values"] = [
                        self._safe_serialize(row[0]) for row in sample_rows
                    ]

                    table_meta["columns"].append(column_meta)

                metadata.append(table_meta)

            return metadata

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Structured schema metadata build failed: {e}")


    def infer_relationships_structured(self):
        """
        Infer many_to_one style relationships using:
        - matching column names
        - matching data types
        - uniqueness heuristics
        """

        try:
            metadata = self.build_structured_schema_metadata()
            relationships = []

            for table_a in metadata:
                for table_b in metadata:

                    if table_a["table_name"] == table_b["table_name"]:
                        continue

                    for col_a in table_a["columns"]:
                        for col_b in table_b["columns"]:

                            # Match by name and type
                            if (
                                col_a["name"] == col_b["name"]
                                and col_a["type"] == col_b["type"]
                            ):

                                # many_to_one detection
                                if col_b.get("unique", False) and not col_a.get("unique", False):

                                    relationships.append({
                                        "from_table": table_a["table_name"],
                                        "from_column": col_a["name"],
                                        "to_table": table_b["table_name"],
                                        "to_column": col_b["name"],
                                        "relationship_type": "many_to_one"
                                    })

            return relationships

        except Exception as e:
            raise RuntimeError(f"[DuckDB] Structured relationship inference failed: {e}")
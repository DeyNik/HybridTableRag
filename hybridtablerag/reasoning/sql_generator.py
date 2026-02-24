import json
from typing import List, Dict, Any
import re
from hybridtablerag.llm.base import BaseLLM

FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE"
]

class SQLValidator:

    @staticmethod
    def validate(sql: str) -> None:
        sql_upper = sql.upper()

        if not sql_upper.strip().startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed.")

        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(f"Forbidden SQL keyword detected: {keyword}")
            
class LLMSQLGenerator:
    """
    LLM SQL generation layer using Ollama (local).
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _build_prompt(
    self,
    user_query: str,
    schema_metadata,
    relationships,
    reasoning: bool = False
    ) -> str:

        if reasoning:
            output_instruction = """
            OUTPUT FORMAT:
            Return strictly valid JSON with this structure:

            {
            "reasoning": "Step by step reasoning explaining how you derived the query.",
            "sql_query": "Valid DuckDB SQL query"
            }

            Do not include markdown.
            Do not include extra text.
            Return only valid JSON.
            """
        else:
            output_instruction = """
            OUTPUT FORMAT:
            Return ONLY the SQL query.
            Do not include markdown.
            Do not include explanations.
            Do not prefix with 'sql'.
            """

        base_instruction = f"""
        You are a senior data engineer.

        Generate a valid DuckDB SQL query.

        STRICT RULES:
        - Use ONLY the provided tables and columns.
        - Do NOT invent tables.
        - Do NOT invent columns.
        - Use GROUP BY when aggregation is required.

        {output_instruction}
        """

        metadata_json = json.dumps(schema_metadata, indent=2)

        prompt = f"""
        {base_instruction}

        User Question:
        {user_query}

        Available Database Metadata:
        {metadata_json}
        """

        # Include relationships only if meaningful
        if len(schema_metadata) > 1 and relationships:
            relationships_json = json.dumps(relationships, indent=2)

            prompt += f"""

            Inferred Relationships:
            {relationships_json}

            Use these relationships for JOIN conditions if needed.
            """

        prompt += "\nGenerate the SQL query:\n"

        return prompt


    def _basic_sql_validation(
        self,
        sql: str,
        schema_metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Basic guardrail to ensure known table usage.
        """

        known_tables = {table["table_name"] for table in schema_metadata}
        sql_lower = sql.lower()

        if not any(table.lower() in sql_lower for table in known_tables):
            raise ValueError("Generated SQL does not reference known tables.")

        return True


    def generate_sql(
    self,
    user_query: str,
    schema_metadata,
    relationships,
    reasoning: bool = False
    ):
        prompt = self._build_prompt(
            user_query=user_query,
            schema_metadata=schema_metadata,
            relationships=relationships,
            reasoning=reasoning
        )

        raw_sql = self.llm.generate(prompt).strip().replace("```","").replace('json','')

        if reasoning:
            try:
                parsed = json.loads(raw_sql)
                sql = parsed["sql_query"]
                explanation = parsed["reasoning"]

                self._basic_sql_validation(sql, schema_metadata)

                return {
                    "sql_query": sql,
                    "reasoning": explanation
                }

            except Exception:
                raise ValueError(f"Invalid JSON response:\n{raw_sql}")

        else:
            sql = raw_sql

            # Clean markdown
            sql = re.sub(r"^```.*?\n", "", sql, flags=re.DOTALL)
            sql = re.sub(r"```$", "", sql).strip()

            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()

            self._basic_sql_validation(sql, schema_metadata)

            return sql
        
       
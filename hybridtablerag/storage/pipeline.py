from hybridtablerag.storage.duckdb_manager import DuckDBManager
from hybridtablerag.reasoning.sql_generator import LLMSQLGenerator

# Initialize database
db = DuckDBManager()

# Build structured metadata
schema_metadata = db.build_structured_schema_metadata()
relationships = db.infer_relationships_structured()

# Initialize LLM generator (Ollama local)
generator = LLMSQLGenerator(model_name="qwen3:8b")

# Natural language query
user_query = "Which industry generated highest revenue?"

# Generate SQL
sql = generator.generate_sql(
    user_query=user_query,
    schema_metadata=schema_metadata,
    relationships=relationships
)

print("\nGenerated SQL:\n")
print(sql)

# Execute SQL
result = db.execute_query(sql)

print("\nQuery Result:\n")
print(result)
"""
storage/utils.py
================
Shared utilities for DuckDB operations.
"""

def _escape_identifier(name: str) -> str:
    """
    Safely escape DuckDB table/column names to prevent SQL injection.
    Usage: _escape_identifier('table"name') -> '"table""name"'
    """
    if not name:
        raise ValueError("Identifier cannot be empty")
    return f'"{name.replace('"', '""')}"'
import os
import pandas as pd


class DatabricksConfigError(RuntimeError):
    pass


def get_databricks_connection():
    host = (os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST") or "").strip().rstrip("/")
    http_path = (os.getenv("DATABRICKS_HTTP_PATH") or "").strip()
    token = (os.getenv("DATABRICKS_ACCESS_TOKEN") or os.getenv("DATABRICKS_TOKEN") or "").strip()

    if not host or not http_path or not token:
        raise DatabricksConfigError("Missing Databricks credentials (hostname/http_path/token)")

    from databricks import sql
    return sql.connect(server_hostname=host, http_path=http_path, access_token=token)


import re

def run_query(connection, query, parameters=None):
    cleaned = (query or "").strip().rstrip(";")
    
    # ------------------ SAFETY CHECK ------------------
    # Strip string literals safely
    q_no_strings = re.sub(r"'(?:\\.|[^'])*'", "", cleaned)
    q_no_strings = re.sub(r'"(?:\\.|[^"])*"', "", q_no_strings)
    # Strip block comments and inline comments
    q_clean = re.sub(r"/\*.*?\*/", "", q_no_strings, flags=re.DOTALL)
    q_clean = re.sub(r"--.*$", "", q_clean, flags=re.MULTILINE)
    
    allowed_starts = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}
    statements = [s.strip().upper() for s in q_clean.split(";") if s.strip()]
    
    for stmt in statements:
        first_word = stmt.split()[0] if stmt.split() else ""
        if first_word and first_word not in allowed_starts:
            raise ValueError(f"Only read-only Databricks queries are allowed. Invalid command: {first_word}")
    # --------------------------------------------------

    with connection.cursor() as cur:
        if parameters:
            cur.execute(cleaned, parameters)
        else:
            cur.execute(cleaned)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
    return cols, rows


def run_readonly_query(connection, query, parameters=None):
    cleaned = (query or "").strip()
    upper_sql = cleaned.upper()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        raise ValueError("Only SELECT/WITH queries are allowed")
    return run_query(connection, cleaned, parameters=parameters)


def fetch_dataframe(connection, query, parameters=None, readonly=False):
    if readonly:
        cols, rows = run_readonly_query(connection, query, parameters=parameters)
    else:
        cols, rows = run_query(connection, query, parameters=parameters)
    return pd.DataFrame(rows, columns=cols)

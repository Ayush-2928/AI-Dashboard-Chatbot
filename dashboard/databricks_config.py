import os


def is_truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_databricks_mode_active():
    return is_truthy(os.getenv("USE_DATABRICKS", "false"))


def databricks_metadata_only_enabled():
    return is_truthy(os.getenv("DATABRICKS_METADATA_ONLY", "true"))


def strict_sql_guardrails_enabled():
    return is_truthy(os.getenv("STRICT_SQL_GUARDRAILS", "true"))


def ai_sql_max_limit(default=5000):
    raw = os.getenv("AI_SQL_MAX_LIMIT", str(default))
    try:
        value = int(raw)
        return max(1, value)
    except Exception:
        return default


def llm_include_sample_rows():
    if is_databricks_mode_active() and databricks_metadata_only_enabled():
        return False
    return True


def databricks_catalog():
    return (os.getenv("DATABRICKS_CATALOG") or "").strip()


def databricks_schema():
    return (os.getenv("DATABRICKS_SCHEMA") or "").strip()


def databricks_source_table():
    explicit = (
        os.getenv("DATABRICKS_SOURCE_TABLE")
        or os.getenv("DATABRICKS_TABLE")
        or os.getenv("DATABRICKS_SOURCE_VIEW")
        or ""
    ).strip()
    if explicit:
        return explicit

    default_table = (os.getenv("DATABRICKS_DEFAULT_TABLE") or "").strip()
    catalog = databricks_catalog()
    schema = databricks_schema()
    if catalog and schema and default_table:
        return f"{catalog}.{schema}.{default_table}"

    return ""



def databricks_source_table_pattern():
    return (os.getenv("DATABRICKS_SOURCE_TABLE_PATTERN") or "").strip()



def databricks_prefer_fact_tables():
    return is_truthy(os.getenv("DATABRICKS_PREFER_FACT_TABLES", "true"))

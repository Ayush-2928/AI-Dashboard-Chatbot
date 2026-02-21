import pandas as pd
import json
import os
import re
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from django.http import HttpResponse
import zipfile
import io
import os
from django.conf import settings
import uuid

from .databricks_config import (
    is_truthy as config_is_truthy,
    is_databricks_mode_active as config_is_databricks_mode_active,
    llm_include_sample_rows as config_llm_include_sample_rows,
    strict_sql_guardrails_enabled as config_strict_sql_guardrails_enabled,
    ai_sql_max_limit as config_ai_sql_max_limit,
    databricks_catalog as config_databricks_catalog,
    databricks_schema as config_databricks_schema,
    databricks_source_table as config_databricks_source_table,
    databricks_source_table_pattern as config_databricks_source_table_pattern,
    databricks_prefer_fact_tables as config_databricks_prefer_fact_tables,
)
from .sql_guardrails import (
    redact_sensitive_text as guardrail_redact_sensitive_text,
    apply_sql_security_and_cost_guardrails as guardrail_apply_sql_security_and_cost_guardrails,
)
from .llm_context_builder import (
    get_table_schema as context_get_table_schema,
    build_schema_context_from_columns,
)
from .databricks_client import (
    get_databricks_connection,
    fetch_dataframe,
    DatabricksConfigError,
)

# --- CONFIGURATION ---
# Always load the project .env (and override stale shell/session vars)
_dotenv_path = os.path.join(str(settings.BASE_DIR), ".env")
load_dotenv(dotenv_path=_dotenv_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


def _is_truthy(value):
    return config_is_truthy(value)


USE_DATABRICKS = config_is_databricks_mode_active()
DATABRICKS_METADATA_ONLY = not config_llm_include_sample_rows()
STRICT_SQL_GUARDRAILS = config_strict_sql_guardrails_enabled()
AI_SQL_MAX_LIMIT = config_ai_sql_max_limit()

DATABRICKS_LOGICAL_VIEW_NAME = "analysis_view"


def _is_databricks_mode_active():
    return config_is_databricks_mode_active()


def _llm_include_sample_rows():
    return config_llm_include_sample_rows()


def _redact_sensitive_text(text):
    return guardrail_redact_sensitive_text(text)

SUPPORTED_DOMAINS = [
    "Supply Chain",
    "Sales",
    "Retail",
    "Finance",
    "E-commerce",
    "Marketing",
    "Human Resources",
    "Production",
    "CRM",
    "Customer Support",
    "General",
]

FIXED_DASHBOARD_THEME = {
    "color": "#22C55E",
    "gradient": "from-emerald-400 to-green-500",
}

def clean_col_name(col):
    return re.sub(r'[^a-zA-Z0-9]', '_', str(col)).lower().strip('_')

def _log_llm_request(debug_logs, context, model_name, json_mode, messages, max_chars=50000):
    if debug_logs is None:
        return
    try:
        parts = []
        for msg in (messages or []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(f"[{role}]\n{_redact_sensitive_text(content)}")

        payload = "\n\n".join(parts).strip()
        payload_chars = len(payload)
        preview = payload[:max_chars]
        if payload_chars > max_chars:
            preview += f"\n... [TRUNCATED {payload_chars - max_chars} chars]"

        debug_logs.append(
            f"[LLM REQUEST] {context} | model={model_name} | json_mode={json_mode} | "
            f"messages={len(messages or [])} | chars={payload_chars}\n{preview}"
        )
    except Exception as e:
        debug_logs.append(f"[LLM REQUEST] {context} | logging failed: {str(e)}")


def _log_llm_response(debug_logs, context, tokens_used, content, max_chars=50000):
    if debug_logs is None:
        return
    try:
        text = content if content is not None else ""
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False)
        text = _redact_sensitive_text(text)

        payload_chars = len(text)
        preview = text[:max_chars]
        if payload_chars > max_chars:
            preview += f"\n... [TRUNCATED {payload_chars - max_chars} chars]"

        debug_logs.append(
            f"[LLM RESPONSE] {context} | tokens={tokens_used} | chars={payload_chars}\n{preview}"
        )
    except Exception as e:
        debug_logs.append(f"[LLM RESPONSE] {context} | logging failed: {str(e)}")
def call_ai_with_retry(messages, json_mode=False, retries=3, debug_logs=None, context="LLM"):
    model_name = "gpt-4o"
    if not client:
        if debug_logs is not None:
            debug_logs.append(f"[LLM REQUEST] {context} skipped: OPENAI_API_KEY not configured")
        return None, 0

    _log_llm_request(debug_logs, context, model_name, json_mode, messages)

    for attempt in range(retries):
        try:
            kwargs = {"model": model_name, "messages": messages, "temperature": 0}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            res = client.chat.completions.create(**kwargs)
            content = res.choices[0].message.content
            tokens_used = res.usage.total_tokens if hasattr(res, 'usage') else 0
            _log_llm_response(debug_logs, context, tokens_used, content)
            return content, tokens_used
        except Exception as e:
            if debug_logs is not None:
                debug_logs.append(f"[LLM ERROR] {context} attempt {attempt + 1}/{retries}: {str(e)}")
            time.sleep(1)

    return None, 0

def get_table_schema(con, table_name, include_sample=True):
    return context_get_table_schema(con, table_name, include_sample=include_sample)



def _find_col_case_insensitive(columns, candidates):
    lower_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _quote_identifier(identifier):
    return "`" + str(identifier).replace("`", "``") + "`"


def _is_numeric_dtype(dtype):
    d = str(dtype).lower()
    return any(t in d for t in ["int", "double", "decimal", "float", "numeric", "bigint", "smallint", "tinyint", "long", "real"])


def _is_text_dtype(dtype):
    d = str(dtype).lower()
    return any(t in d for t in ["string", "varchar", "char", "text"])


def _is_date_dtype(dtype):
    d = str(dtype).lower()
    return "date" in d or "timestamp" in d or "time" in d


def _score_databricks_table_name(table_name, prefer_fact=True):
    name = str(table_name or "").strip().lower()
    if not name:
        return -10_000

    score = 0

    if prefer_fact:
        if re.search(r"(^fact_|_fact_|fact$|invoice|sales|txn|transaction|order|billing|receipt)", name):
            score += 120
        if name.startswith("fact_"):
            score += 40

    if re.search(r"(^dim_|_dim_|dimension|master)", name):
        score -= 35

    if re.search(r"(tmp|temp|stg|staging|backup|history|hist|snapshot)", name):
        score -= 25

    return score


def _resolve_databricks_source_table(connection, logs=None):
    configured = config_databricks_source_table()
    if configured:
        if logs is not None:
            logs.append(f"[SOURCE] Using configured Databricks source: {configured}")
        return configured

    catalog = config_databricks_catalog()
    schema = config_databricks_schema()
    if not catalog or not schema:
        raise DatabricksConfigError(
            "Set DATABRICKS_SOURCE_TABLE or provide DATABRICKS_CATALOG + DATABRICKS_SCHEMA"
        )

    tables_df = fetch_dataframe(connection, f"SHOW TABLES IN {_quote_identifier(catalog)}.{_quote_identifier(schema)}")
    if tables_df.empty:
        raise DatabricksConfigError(f"No tables/views found in {catalog}.{schema}")

    table_col = _find_col_case_insensitive(tables_df.columns, ["tableName", "table_name", "tablename"])
    db_col = _find_col_case_insensitive(tables_df.columns, ["database", "schema", "namespace"])
    if not table_col:
        raise DatabricksConfigError("Unable to parse SHOW TABLES output")

    pattern_text = config_databricks_source_table_pattern()
    prefer_fact = config_databricks_prefer_fact_tables()
    pattern = None
    if pattern_text:
        try:
            pattern = re.compile(pattern_text, flags=re.IGNORECASE)
        except re.error:
            if logs is not None:
                logs.append(f"[WARN] Invalid DATABRICKS_SOURCE_TABLE_PATTERN ignored: {pattern_text}")

    ranked = []
    for _, row in tables_df.iterrows():
        table_name = str(row.get(table_col, "") or "").strip()
        if not table_name:
            continue

        db_name = str(row.get(db_col, "") or "").strip() if db_col else schema
        score = _score_databricks_table_name(table_name, prefer_fact=prefer_fact)

        if pattern and pattern.search(table_name):
            score += 1000

        ranked.append((score, table_name.lower(), table_name, db_name or schema))

    if not ranked:
        raise DatabricksConfigError("Could not resolve a source table from Databricks schema")

    ranked.sort(key=lambda x: (-x[0], x[1]))
    _, _, selected_table, selected_db = ranked[0]
    selected_fqn = (
        f"{_quote_identifier(catalog)}."
        f"{_quote_identifier(selected_db)}."
        f"{_quote_identifier(selected_table)}"
    )

    if logs is not None:
        logs.append(
            f"[SOURCE] Auto-selected Databricks source: {selected_fqn} "
            f"(prefer_fact={prefer_fact}, pattern={'set' if pattern else 'none'})"
        )

    return selected_fqn


def _describe_databricks_table_columns(connection, table_name):
    describe_df = fetch_dataframe(connection, f"DESCRIBE {table_name}")
    if describe_df.empty:
        return []

    col_name_key = _find_col_case_insensitive(describe_df.columns, ["col_name", "column_name"])
    col_type_key = _find_col_case_insensitive(describe_df.columns, ["data_type", "column_type", "type"])
    if not col_name_key or not col_type_key:
        if len(describe_df.columns) < 2:
            return []
        col_name_key, col_type_key = describe_df.columns[0], describe_df.columns[1]

    columns = []
    for _, row in describe_df.iterrows():
        col_name = str(row[col_name_key]).strip()
        if not col_name or col_name.startswith("#"):
            continue
        col_type = str(row[col_type_key]).strip() or "STRING"
        columns.append((col_name, col_type))
    return columns


def _load_databricks_schema_context(connection, source_table, include_sample_rows):
    schema_columns = _describe_databricks_table_columns(connection, source_table)
    if not schema_columns:
        raise ValueError(f"DESCRIBE returned no schema for {source_table}")

    sample_df = None
    if include_sample_rows:
        sample_df = fetch_dataframe(connection, f"SELECT * FROM {source_table} LIMIT 2", readonly=True)

    context = build_schema_context_from_columns(
        DATABRICKS_LOGICAL_VIEW_NAME,
        schema_columns,
        include_sample=include_sample_rows,
        sample_df=sample_df,
    )
    return context, schema_columns


def _safe_int_env(name, default):
    raw = os.getenv(name, str(default))
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _databricks_query_row_cap():
    raw = os.getenv("DATABRICKS_QUERY_ROW_CAP", "0")
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def _unquote_databricks_identifier(name):
    return str(name or "").replace("`", "").strip()

def _load_databricks_schema_context_from_query_source(connection, query_source, include_sample_rows, schema_columns):
    sample_df = None
    if include_sample_rows:
        sample_df = fetch_dataframe(connection, f"SELECT * FROM {query_source} LIMIT 2", readonly=True)
    return build_schema_context_from_columns(
        DATABRICKS_LOGICAL_VIEW_NAME,
        schema_columns,
        include_sample=include_sample_rows,
        sample_df=sample_df,
    )


def _build_databricks_virtual_source(connection, include_sample_rows, logs=None):
    base_table = _resolve_databricks_source_table(connection, logs=logs)
    base_columns = _describe_databricks_table_columns(connection, base_table)
    if not base_columns:
        raise ValueError(f"No columns found for Databricks source {base_table}")

    schema_context = _load_databricks_schema_context_from_query_source(
        connection,
        base_table,
        include_sample_rows,
        base_columns,
    )
    if logs is not None:
        logs.append("[SOURCE] Databricks using source table only (auto-join removed)")

    return {
        "base_table": base_table,
        "query_source": base_table,
        "schema_columns": base_columns,
        "schema_context": schema_context,
        "joined_tables": [],
    }


def _build_databricks_where_clause(active_filters_json, available_columns, date_column=None):
    if not active_filters_json:
        return "", 0

    try:
        filters = json.loads(active_filters_json)
    except Exception:
        return "", 0

    available = {str(c).lower(): c for c in available_columns}
    where_clauses = []

    start_date = filters.get("_start_date")
    end_date = filters.get("_end_date")
    if date_column:
        date_ident = _quote_identifier(date_column)
        if start_date:
            safe = str(start_date).replace("'", "''")
            where_clauses.append(f"CAST({date_ident} AS DATE) >= CAST('{safe}' AS DATE)")
        if end_date:
            safe = str(end_date).replace("'", "''")
            where_clauses.append(f"CAST({date_ident} AS DATE) <= CAST('{safe}' AS DATE)")

    for col, val in filters.items():
        if col in {"_start_date", "_end_date"}:
            continue
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "null":
            continue

        existing = available.get(str(col).lower())
        if not existing:
            continue

        safe_val = str(val).replace("'", "''")
        where_clauses.append(f"{_quote_identifier(existing)} = '{safe_val}'")

    if not where_clauses:
        return "", 0

    return " AND ".join(where_clauses), len(where_clauses)


def _normalize_databricks_sql_references(user_sql, source_table, view_name=DATABRICKS_LOGICAL_VIEW_NAME):
    sql_text = str(user_sql or "")
    source_raw = str(source_table or "").strip()
    source_plain = source_raw.replace("`", "")

    candidates = {"master_view", "final_view", source_raw, source_plain}
    if source_plain:
        tail = source_plain.split(".")[-1]
        candidates.add(tail)
        candidates.add(f"`{tail}`")

    normalized = sql_text
    for cand in sorted([c for c in candidates if c], key=len, reverse=True):
        normalized = re.sub(
            rf"(?i)(?<![A-Za-z0-9_`]){re.escape(cand)}(?![A-Za-z0-9_`])",
            view_name,
            normalized,
        )
    return normalized


def _normalize_databricks_sql_dialect(sql_text):
    sql_value = str(sql_text or "")

    # Databricks SQL expects STRING type; bare VARCHAR without length fails.
    sql_value = re.sub(r"(?i)\bAS\s+VARCHAR\s*(\(\s*\d+\s*\))?", "AS STRING", sql_value)
    sql_value = re.sub(r"(?i)::\s*VARCHAR\s*(\(\s*\d+\s*\))?", "::STRING", sql_value)

    return sql_value


def _wrap_sql_with_virtual_views(user_sql, source_table, where_sql="", view_name=DATABRICKS_LOGICAL_VIEW_NAME, query_source=None):
    cleaned = str(user_sql or "").strip().rstrip(";")
    base_source = query_source if query_source else source_table
    base_select = f"SELECT * FROM {base_source}"
    if where_sql:
        base_select += f" WHERE {where_sql}"

    row_cap = _databricks_query_row_cap()
    if row_cap > 0:
        base_select = f"SELECT * FROM ({base_select}) __base_capped LIMIT {row_cap}"

    return (
        f"WITH {view_name} AS ({base_select}), "
        f"__user_query AS ({cleaned}) "
        f"SELECT * FROM __user_query"
    )


def _execute_databricks_user_sql(connection, user_sql, source_table, where_sql="", logs=None, context="Databricks Query", query_source=None):
    normalized_sql = _normalize_databricks_sql_references(
        user_sql,
        source_table,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
    )
    dialect_sql = _normalize_databricks_sql_dialect(normalized_sql)
    guarded_sql, notes = _apply_sql_security_and_cost_guardrails(dialect_sql)
    if logs is not None:
        if dialect_sql != normalized_sql:
            logs.append(f"[SECURITY] {context}: Normalized SQL types for Databricks dialect (VARCHAR -> STRING)")
        for note in notes:
            logs.append(f"[SECURITY] {context}: {note}")
        row_cap = _databricks_query_row_cap()
        if row_cap > 0:
            logs.append(f"[PERF] {context}: Applying Databricks row cap DATABRICKS_QUERY_ROW_CAP={row_cap}")

    wrapped_sql = _wrap_sql_with_virtual_views(
        guarded_sql,
        source_table,
        where_sql=where_sql,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
        query_source=query_source,
    )
    df = fetch_dataframe(connection, wrapped_sql, readonly=True)
    if df.empty:
        return df, guarded_sql

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(0)
    return df, guarded_sql


def _default_custom_chart_plan_from_columns(schema_columns, user_prompt, table_name="final_view"):
    text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
    num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
    date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]

    wants_line = any(token in user_prompt.lower() for token in ["trend", "line", "over time", "monthly", "daily"])

    if date_cols and num_cols and wants_line:
        d_col = _quote_identifier(date_cols[0])
        n_col = _quote_identifier(num_cols[0])
        return {
            "type": "line",
            "title": f"{num_cols[0].replace('_', ' ').title()} Trend",
            "xlabel": date_cols[0].replace("_", " ").title(),
            "ylabel": num_cols[0].replace("_", " ").title(),
            "sql": f"SELECT CAST({d_col} AS DATE) AS x, SUM({n_col}) AS y FROM {table_name} GROUP BY 1 ORDER BY 1"
        }

    if text_cols and num_cols:
        t_col = _quote_identifier(text_cols[0])
        n_col = _quote_identifier(num_cols[0])
        return {
            "type": "bar",
            "title": f"{num_cols[0].replace('_', ' ').title()} by {text_cols[0].replace('_', ' ').title()}",
            "xlabel": text_cols[0].replace("_", " ").title(),
            "ylabel": num_cols[0].replace("_", " ").title(),
            "sql": f"SELECT CAST({t_col} AS STRING) AS x, SUM({n_col}) AS y FROM {table_name} WHERE {t_col} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
        }

    if text_cols:
        t_col = _quote_identifier(text_cols[0])
        return {
            "type": "bar",
            "title": f"Record Count by {text_cols[0].replace('_', ' ').title()}",
            "xlabel": text_cols[0].replace("_", " ").title(),
            "ylabel": "Count",
            "sql": f"SELECT CAST({t_col} AS STRING) AS x, COUNT(*) AS y FROM {table_name} WHERE {t_col} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
        }

    if num_cols:
        n_col = _quote_identifier(num_cols[0])
        return {
            "type": "line",
            "title": f"{num_cols[0].replace('_', ' ').title()} Sample",
            "xlabel": "Row",
            "ylabel": num_cols[0].replace("_", " ").title(),
            "sql": f"SELECT ROW_NUMBER() OVER () AS x, CAST({n_col} AS DOUBLE) AS y FROM {table_name} WHERE {n_col} IS NOT NULL LIMIT 50"
        }

    return {
        "type": "bar",
        "title": "Record Count",
        "xlabel": "Category",
        "ylabel": "Count",
        "sql": f"SELECT 'All Data' AS x, COUNT(*) AS y FROM {table_name}"
    }


# --- PHASE 1: ARCHITECT ---
def generate_join_sql(raw_schema_context, debug_logs=None):
    prompt = f"""
    You are a Data Architect. DuckDB Schema:
    {raw_schema_context}
    TASK: Write ONE SQL query to create 'master_view'.
    RULES:
    1. Use `clean_id(t1.col) = clean_id(t2.col)` ONLY for IDs (e.g. OrderID, ProductID).
    2. Do NOT use `SELECT *`. Select specific columns with aliases.
    RETURN RAW SQL ONLY.
    """
    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=False, debug_logs=debug_logs, context="Generate Join SQL")
    return (res.replace('```sql', '').replace('```', '').strip() if res else None), tokens

# --- PHASE 2: ANALYST ---
def generate_viz_config(master_schema_context, forced_domain=None, debug_logs=None, logical_table_name="master_view"):
    domains = ", ".join(SUPPORTED_DOMAINS)
    prompt = f"""
    You are a BI Expert. Analyze 'master_view':
    {master_schema_context}
    
    STEP 1: DETECT DOMAIN -> One of [{domains}]
    {f"(HINT: Looks like {forced_domain})" if forced_domain else ""}
    
    STEP 2: FILTERS -> 3 categorical columns for filtering.
    STEP 3: CHARTS -> 6 SQL queries (including heatmap if applicable).

    CRITICAL DATA GRAIN RULE:
    - master_view/final_view is transaction-level (one row per transaction/event).
    - Master entities (supplier/customer/product/employee etc.) can repeat across many rows.
    - Transaction metrics (revenue, quantity, totals, trends) can aggregate directly on master_view.
    - Master attributes (rating, age, salary, static price, static score, etc.) MUST be deduplicated by entity key before AVG or similar stats.
    - For unique entity counts, use COUNT(DISTINCT <detected_entity_key>), not COUNT(*).
    - Dedup pattern example:
      SELECT region, AVG(performance_rating)
      FROM (
          SELECT DISTINCT supplier_id, region, performance_rating
          FROM master_view
      ) d
      GROUP BY region
    
    **MANDATORY CHART RULES**:
    - **Chart 0 (Line/Trend)**: MUST be a trend over time. SQL: `SELECT CAST(date_col AS DATE) as x, SUM(numeric_col) as y FROM master_view GROUP BY 1 ORDER BY 1`
    - **Chart 1 (Heatmap)**: If you have TWO categorical dimensions and a numeric value, create a heatmap. SQL: `SELECT cat1 as x, cat2 as y, SUM(value) as z FROM master_view GROUP BY 1,2`. Otherwise, make it a bar chart.
    - **Chart 2**: No fixed type; choose the best visualization based on the data and question.
    - **Charts 3-5**: Choose the BEST visualization (line for trends, bar for categorical comparisons) based on the data
    - Use LINE charts when showing trends over time periods
    - Use BAR charts when comparing categories or showing rankings
    
     **TITLE REQUIREMENTS**:
    - Titles MUST be specific and describe what metric is being shown in detail
    - BAD: "Monthly Trend", "Distribution", "Breakdown"
    - GOOD: "Monthly Revenue Trend", "Sales by Region", "Product Category Performance"
    - Include the actual metric name (Revenue, Count, Orders, etc.) and dimension (by Month, by Region, etc.)
        
    
    RETURN JSON:
    {{
        "domain": "Supply Chain",
        "filters": ["Region", "Status", "Category"],
        "kpis": [ 
            {{ "label": "Total Spend", "sql": "SELECT SUM(amount) FROM master_view", "trend_sql": "SELECT CAST(date_col AS DATE) as x, SUM(amount) as y FROM master_view GROUP BY 1 ORDER BY 1 LIMIT 7" }},
            {{ "label": "Record Count", "sql": "SELECT COUNT(*) FROM master_view", "trend_sql": "SELECT CAST(date_col AS DATE) as x, COUNT(*) as y FROM master_view GROUP BY 1 ORDER BY 1 LIMIT 7" }}
        ],
        "charts": [ 
            {{ "title": "Monthly Trend", "type": "line", "sql": "SELECT date_col as x, SUM(val) as y FROM master_view GROUP BY 1 ORDER BY 1", "xlabel": "Date", "ylabel": "Amount" }},
            {{ "title": "Heatmap", "type": "heatmap", "sql": "SELECT cat1 as x, cat2 as y, SUM(val) as z FROM master_view GROUP BY 1,2", "xlabel": "Category 1", "ylabel": "Category 2" }},
            {{ "title": "Top Categories by Value", "type": "bar", "sql": "SELECT CAST(cat1 AS VARCHAR) AS x, SUM(val) AS y FROM master_view GROUP BY 1 ORDER BY 2 DESC LIMIT 12", "xlabel": "Category", "ylabel": "Value" }}
        ]
    }}
    """
    if logical_table_name != "master_view":
        prompt = prompt.replace("master_view/final_view", logical_table_name)
        prompt = prompt.replace("master_view", logical_table_name)
        prompt = prompt.replace("final_view", logical_table_name)

    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=True, debug_logs=debug_logs, context="Generate Viz Config")
    return (json.loads(res) if res else None), tokens

ALLOWED_CUSTOM_CHART_TYPES = {"bar", "line", "scatter", "pie", "heatmap", "table", "treemap", "waterfall"}
FORBIDDEN_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "ATTACH", "DETACH", "COPY", "PRAGMA", "CALL",
    "EXPORT", "IMPORT"
}
SQL_RESERVED_WORDS = {
    "where", "group", "order", "limit", "having", "qualify", "join", "inner",
    "left", "right", "full", "cross", "union", "intersect", "except", "on"
}
AVG_COL_REGEX = re.compile(
    r"\bavg\s*\(\s*(?:cast\s*\(\s*)?(?:(?:[a-zA-Z_][a-zA-Z0-9_]*)\.)?(?P<col>[a-zA-Z_][a-zA-Z0-9_]*)",
    flags=re.IGNORECASE
)


def _looks_like_entity_key(col_name):
    c = col_name.lower()
    return bool(re.search(r"(^id$|_id$|^id_|_id_|^code$|_code$|^key$|_key$|uuid)", c))


def build_entity_dedup_profile(con, table_name="final_view", max_key_cols=8, max_numeric_cols=20):
    profile = {
        "table": table_name,
        "row_count": 0,
        "candidate_keys": [],
        "attribute_to_key": {}
    }
    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0
    except Exception:
        return profile

    profile["row_count"] = int(row_count)
    if row_count <= 1:
        return profile

    try:
        schema = con.execute(f"DESCRIBE {table_name}").df()
    except Exception:
        return profile

    numeric_types = ("INT", "DOUBLE", "DECIMAL", "FLOAT", "NUMERIC", "REAL", "BIGINT", "HUGEINT")
    candidate_keys = []
    numeric_cols = []

    for _, row in schema.iterrows():
        col = row["column_name"]
        dtype = str(row["column_type"]).upper()

        if any(t in dtype for t in numeric_types):
            numeric_cols.append(col)

        try:
            distinct_count = con.execute(
                f"SELECT COUNT(DISTINCT {col}) FROM {table_name} WHERE {col} IS NOT NULL"
            ).fetchone()[0] or 0
        except Exception:
            continue

        if distinct_count < 2:
            continue

        duplicate_factor = row_count / max(distinct_count, 1)
        distinct_ratio = distinct_count / max(row_count, 1)
        key_like = _looks_like_entity_key(col)

        if duplicate_factor >= 1.2 and (key_like or distinct_ratio >= 0.02):
            candidate_keys.append({
                "col": col,
                "distinct_count": distinct_count,
                "duplicate_factor": duplicate_factor,
                "key_like": key_like
            })

    candidate_keys.sort(
        key=lambda x: (x["key_like"], x["distinct_count"], x["duplicate_factor"]),
        reverse=True
    )
    candidate_keys = candidate_keys[:max_key_cols]
    profile["candidate_keys"] = [k["col"] for k in candidate_keys]

    if not candidate_keys:
        return profile

    numeric_cols = [c for c in numeric_cols if c not in profile["candidate_keys"]]
    numeric_cols = numeric_cols[:max_numeric_cols]

    min_entities = max(10, int(row_count * 0.01))
    for attr_col in numeric_cols:
        best_key = None
        best_score = 0

        for key_info in candidate_keys:
            key_col = key_info["col"]
            try:
                entity_count, stable_ratio = con.execute(
                    f"""
                    WITH per_entity AS (
                        SELECT
                            {key_col} AS entity_key,
                            COUNT(DISTINCT ROUND(CAST({attr_col} AS DOUBLE), 6)) AS value_variants
                        FROM {table_name}
                        WHERE {key_col} IS NOT NULL AND {attr_col} IS NOT NULL
                        GROUP BY 1
                    )
                    SELECT
                        COUNT(*) AS entity_count,
                        AVG(CASE WHEN value_variants <= 1 THEN 1 ELSE 0 END) AS stable_ratio
                    FROM per_entity
                    """
                ).fetchone()
            except Exception:
                continue

            entity_count = int(entity_count or 0)
            stable_ratio = float(stable_ratio or 0.0)
            if entity_count < min_entities:
                continue

            score = stable_ratio * min(1.0, entity_count / max(row_count * 0.4, 1))
            if score > best_score:
                best_score = score
                best_key = key_col

        if best_key and best_score >= 0.75:
            profile["attribute_to_key"][attr_col.lower()] = best_key

    return profile


def _rewrite_sql_for_master_attribute_dedup(sql, dedup_profile, table_name="final_view"):
    if not sql or not dedup_profile:
        return sql, None

    attr_to_key = dedup_profile.get("attribute_to_key", {})
    if not attr_to_key:
        return sql, None

    lower_sql = sql.lower()
    if not re.search(rf"\bfrom\s+{re.escape(table_name.lower())}\b", lower_sql):
        return sql, None

    if re.search(r"\bsum\s*\(", lower_sql) or re.search(r"\bcount\s*\(\s*(\*|1)\s*\)", lower_sql):
        return sql, None

    avg_cols = [m.group("col").lower() for m in AVG_COL_REGEX.finditer(sql)]
    if not avg_cols:
        return sql, None

    mapped_keys = []
    mapped_cols = []
    for col in avg_cols:
        key_col = attr_to_key.get(col)
        if not key_col:
            return sql, None
        mapped_keys.append(key_col)
        mapped_cols.append(col)

    if len(set(mapped_keys)) != 1:
        return sql, None

    entity_key = mapped_keys[0]
    from_match = re.search(
        rf"\bfrom\s+{re.escape(table_name)}\b",
        sql,
        flags=re.IGNORECASE
    )
    if not from_match:
        return sql, None

    alias = None
    replace_end = from_match.end()
    alias_match = re.search(
        rf"\bfrom\s+{re.escape(table_name)}\s+(?:as\s+)?(?P<alias>[a-zA-Z_][a-zA-Z0-9_]*)\b",
        sql,
        flags=re.IGNORECASE
    )
    if alias_match and alias_match.start() == from_match.start():
        candidate_alias = alias_match.group("alias")
        if candidate_alias and candidate_alias.lower() not in SQL_RESERVED_WORDS:
            alias = candidate_alias
            replace_end = alias_match.end()

    dedup_alias = alias or table_name
    dedup_source = (
        f"(SELECT * FROM {table_name} "
        f"WHERE {entity_key} IS NOT NULL "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {entity_key}) = 1)"
    )
    replacement = f"FROM {dedup_source} AS {dedup_alias}"
    rewritten = sql[:from_match.start()] + replacement + sql[replace_end:]
    if rewritten == sql:
        return sql, None

    note = (
        f"Applied entity dedup using key '{entity_key}' for AVG on "
        f"{', '.join(sorted(set(mapped_cols)))}"
    )
    return rewritten, note


def execute_sql_with_dedup(con, sql, dedup_profile=None, logs=None, context="Query", table_name="final_view"):
    rewritten_sql, note = _rewrite_sql_for_master_attribute_dedup(
        sql, dedup_profile, table_name=table_name
    )
    if note and logs is not None:
        logs.append(f"[DEDUP] {context}: {note}")
    return con.execute(rewritten_sql), rewritten_sql

def _apply_sql_security_and_cost_guardrails(sql, max_limit=AI_SQL_MAX_LIMIT):
    return guardrail_apply_sql_security_and_cost_guardrails(
        sql,
        forbidden_sql_keywords=FORBIDDEN_SQL_KEYWORDS,
        databricks_mode=_is_databricks_mode_active(),
        strict_guardrails=STRICT_SQL_GUARDRAILS,
        max_limit=max_limit,
    )


def _safe_custom_sql(sql):
    if not sql:
        return False

    cleaned = sql.replace("```sql", "").replace("```", "").strip().rstrip(";")
    upper_sql = cleaned.upper()

    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        return False

    if "FINAL_VIEW" not in upper_sql and "MASTER_VIEW" not in upper_sql and DATABRICKS_LOGICAL_VIEW_NAME.upper() not in upper_sql:
        return False

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False

    return True

def _default_custom_chart_plan(con, user_prompt):
    schema = con.execute("DESCRIBE final_view").df()
    text_cols = []
    num_cols = []
    date_cols = []

    for _, row in schema.iterrows():
        col = row["column_name"]
        dtype = str(row["column_type"]).upper()
        if "VARCHAR" in dtype or "TEXT" in dtype:
            text_cols.append(col)
        if any(t in dtype for t in ["INT", "DOUBLE", "DECIMAL", "FLOAT", "NUMERIC", "BIGINT"]):
            num_cols.append(col)
        if "DATE" in dtype or "TIMESTAMP" in dtype:
            date_cols.append(col)

    wants_line = any(token in user_prompt.lower() for token in ["trend", "line", "over time", "monthly", "daily"])

    if date_cols and num_cols and wants_line:
        d_col = date_cols[0]
        n_col = num_cols[0]
        return {
            "type": "line",
            "title": f"{n_col.replace('_', ' ').title()} Trend",
            "xlabel": d_col.replace("_", " ").title(),
            "ylabel": n_col.replace("_", " ").title(),
            "sql": f"SELECT CAST({d_col} AS DATE) AS x, SUM({n_col}) AS y FROM final_view GROUP BY 1 ORDER BY 1"
        }

    if text_cols and num_cols:
        t_col = text_cols[0]
        n_col = num_cols[0]
        return {
            "type": "bar",
            "title": f"{n_col.replace('_', ' ').title()} by {t_col.replace('_', ' ').title()}",
            "xlabel": t_col.replace("_", " ").title(),
            "ylabel": n_col.replace("_", " ").title(),
            "sql": f"SELECT CAST({t_col} AS VARCHAR) AS x, SUM({n_col}) AS y FROM final_view WHERE {t_col} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
        }

    if text_cols:
        t_col = text_cols[0]
        return {
            "type": "bar",
            "title": f"Record Count by {t_col.replace('_', ' ').title()}",
            "xlabel": t_col.replace("_", " ").title(),
            "ylabel": "Count",
            "sql": f"SELECT CAST({t_col} AS VARCHAR) AS x, COUNT(*) AS y FROM final_view WHERE {t_col} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
        }

    if num_cols:
        n_col = num_cols[0]
        return {
            "type": "line",
            "title": f"{n_col.replace('_', ' ').title()} Sample",
            "xlabel": "Row",
            "ylabel": n_col.replace("_", " ").title(),
            "sql": f"SELECT ROW_NUMBER() OVER () AS x, CAST({n_col} AS DOUBLE) AS y FROM final_view WHERE {n_col} IS NOT NULL LIMIT 50"
        }

    return {
        "type": "bar",
        "title": "Record Count",
        "xlabel": "Category",
        "ylabel": "Count",
        "sql": "SELECT 'All Data' AS x, COUNT(*) AS y FROM final_view"
    }

def generate_custom_chart_plan(final_schema_context, user_prompt, debug_logs=None, table_name="final_view"):
    prompt = f"""
    You are a senior BI analyst. Build one chart spec from a natural language request.

    DATA SCHEMA (DuckDB table: final_view):
    {final_schema_context}

    USER REQUEST:
    {user_prompt}

    RULES:
    1. Return ONLY JSON.
    2. Supported chart types: bar, line, scatter, pie, heatmap, table, treemap, waterfall.
    3. SQL must be a SELECT/WITH query using only final_view.
    4. Alias output columns exactly:
       - bar/line/scatter/pie: x, y
       - heatmap: x, y, z
       - table: keep meaningful business column names (do NOT alias to generic columns/rows)
       - treemap: labels, parents, values
       - waterfall: x, y (optional measure)
    5. For category charts, include LIMIT 30 or less.
    6. Use safe casts where useful (CAST(col AS VARCHAR) for categories, CAST(date AS DATE) for dates).
    7. final_view is transaction-level. For transactional metrics use final_view directly.
    8. For master/entity attributes (rating, age, salary, static price, scores, etc.), deduplicate by entity key before AVG-style aggregations.
    9. For unique entity questions, use COUNT(DISTINCT <detected_entity_key>) instead of COUNT(*).

    JSON format:
    {{
      "title": "Sales by Region",
      "type": "bar",
      "sql": "SELECT CAST(region AS VARCHAR) AS x, SUM(sales) AS y FROM final_view GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
      "xlabel": "Region",
      "ylabel": "Sales"
    }}
    """
    if table_name != "final_view":
        prompt = prompt.replace("final_view", table_name)

    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=True, debug_logs=debug_logs, context="Generate Custom Chart Plan")
    try:
        parsed = json.loads(res) if res else None
    except Exception:
        parsed = None
    return parsed, tokens



CUSTOM_CHART_DISAMBIGUATION_STOPWORDS = {
    "show", "plot", "chart", "graph", "visual", "visualize", "give", "create", "generate",
    "for", "from", "with", "using", "into", "onto", "by", "of", "to", "in", "on", "at",
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "this", "that",
    "month", "months", "weekly", "daily", "year", "years", "trend", "total", "count",
    "sum", "average", "avg", "min", "max", "top", "bottom", "highest", "lowest",
    "sales", "revenue", "invoice", "invoices", "amount", "quantity", "value", "values",
    "net", "gross", "bill", "billing", "monthwise", "mom",
}


def _extract_custom_chart_disambiguation_terms(user_prompt, max_terms=8):
    prompt = str(user_prompt or "").strip().lower()
    if not prompt:
        return [], []

    quoted_terms = []
    for q in re.findall(r'["\']([^"\']{2,80})["\']', prompt):
        q_clean = re.sub(r"\s+", " ", q.strip())
        if len(q_clean) >= 2:
            quoted_terms.append(q_clean)

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", prompt)
    words = [w for w in words if w not in CUSTOM_CHART_DISAMBIGUATION_STOPWORDS]

    single_terms = []
    for w in words:
        if w not in single_terms:
            single_terms.append(w)

    bigrams = []
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in bigrams:
            bigrams.append(bg)

    column_terms = []
    for term in quoted_terms + bigrams + single_terms:
        if term not in column_terms:
            column_terms.append(term)

    value_terms = []
    for term in quoted_terms + bigrams:
        if len(term) >= 3 and term not in value_terms:
            value_terms.append(term)

    return column_terms[:max_terms], value_terms[:max_terms]


def _column_name_matches_term(column_name, term):
    col = str(column_name or "").strip().lower()
    t = str(term or "").strip().lower()
    if not col or not t:
        return False

    if t in col:
        return True

    col_tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", col) if tok]
    term_tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", t) if tok]
    if not col_tokens or not term_tokens:
        return False

    if len(term_tokens) == 1:
        return term_tokens[0] in col_tokens

    return all(tok in col_tokens for tok in term_tokens)


def _rank_text_columns_for_disambiguation(schema_columns, max_columns=14):
    priority_tokens = [
        "name", "product", "brand", "category", "type", "status", "region", "city",
        "state", "country", "channel", "group", "segment", "market", "customer",
    ]

    text_columns = [col for col, dtype in schema_columns if _is_text_dtype(dtype)]
    ranked = []
    for col in text_columns:
        lower = str(col).lower()
        score = 0
        for idx, tok in enumerate(priority_tokens):
            if tok in lower:
                score += max(1, 20 - idx)
        if lower.endswith("_id") or lower == "id":
            score -= 8
        ranked.append((score, lower, col))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [col for _, _, col in ranked[:max_columns]]


def _column_contains_value_term(connection, source_query, where_sql, column_name, value_term):
    term = str(value_term or "").strip().lower()
    if len(term) < 2:
        return False

    safe_term = term.replace("'", "''")
    col_ident = _quote_identifier(column_name)
    where_parts = []
    if where_sql:
        where_parts.append(f"({where_sql})")
    where_parts.append(f"{col_ident} IS NOT NULL")
    where_parts.append(f"LOWER(CAST({col_ident} AS STRING)) LIKE '%{safe_term}%'")

    sql = f"SELECT 1 AS hit FROM {source_query} WHERE {' AND '.join(where_parts)} LIMIT 1"
    try:
        df = fetch_dataframe(connection, sql, readonly=True)
        return not df.empty
    except Exception:
        return False


def _normalize_custom_chart_clarification_choice(choice, schema_columns):
    if not isinstance(choice, dict):
        return None

    keyword = str(choice.get("keyword") or "").strip().lower()
    column_raw = str(choice.get("column") or "").strip()
    if not keyword and not column_raw:
        return None

    available_map = {str(c).lower(): c for c, _ in schema_columns}
    normalized_column = available_map.get(column_raw.lower()) if column_raw else None

    return {
        "keyword": keyword,
        "column": normalized_column,
    }


def _apply_custom_chart_clarification_to_prompt(user_prompt, clarification_choice, logs=None):
    if not clarification_choice:
        return user_prompt

    keyword = str(clarification_choice.get("keyword") or "").strip()
    column = str(clarification_choice.get("column") or "").strip()
    if not keyword or not column:
        return user_prompt

    if logs is not None:
        logs.append(f"[CLARIFY] User selected column '{column}' for keyword '{keyword}'")

    addition = (
        "\n\nUSER CLARIFICATION:\n"
        f"- Interpret '{keyword}' using column '{column}'.\n"
        "- Use this mapping consistently in SQL and labels."
    )
    return str(user_prompt or "") + addition


def _build_custom_chart_clarification(keyword, candidate_columns, reason):
    options = []
    for col in candidate_columns[:8]:
        options.append({
            "column": col,
            "label": col,
        })

    reason_text = "column name" if reason == "column_name" else "value"
    return {
        "keyword": keyword,
        "reason": reason,
        "question": (
            f"The term '{keyword}' matches multiple columns by {reason_text}. "
            "Please choose the correct column."
        ),
        "options": options,
    }


def _detect_custom_chart_ambiguity(
    connection,
    user_prompt,
    schema_columns,
    source_query,
    where_sql="",
    clarification_choice=None,
    logs=None,
):
    column_terms, value_terms = _extract_custom_chart_disambiguation_terms(user_prompt)
    if not column_terms and not value_terms:
        return None

    chosen_keyword = ""
    chosen_column = ""
    if clarification_choice:
        chosen_keyword = str(clarification_choice.get("keyword") or "").strip().lower()
        chosen_column = str(clarification_choice.get("column") or "").strip()

    columns_only = [c for c, _ in schema_columns]

    # Pass 1: keyword-to-column name ambiguity.
    for term in column_terms:
        term_l = str(term).strip().lower()
        if not term_l:
            continue
        if chosen_keyword and term_l == chosen_keyword and chosen_column:
            continue

        matches = [c for c in columns_only if _column_name_matches_term(c, term_l)]
        matches = list(dict.fromkeys(matches))
        if len(matches) >= 2:
            if logs is not None:
                logs.append(
                    f"[CLARIFY] Ambiguous keyword '{term_l}' matched columns: {', '.join(matches[:6])}"
                )
            return _build_custom_chart_clarification(term_l, matches, reason="column_name")

    # Pass 2: value appears in multiple text columns.
    ranked_text_cols = _rank_text_columns_for_disambiguation(schema_columns)
    if not ranked_text_cols:
        return None

    for term in value_terms:
        term_l = str(term).strip().lower()
        if len(term_l) < 3:
            continue
        if chosen_keyword and term_l == chosen_keyword and chosen_column:
            continue

        matches = []
        for col in ranked_text_cols:
            if _column_contains_value_term(connection, source_query, where_sql, col, term_l):
                matches.append(col)
            if len(matches) >= 6:
                break

        matches = list(dict.fromkeys(matches))
        if len(matches) >= 2:
            if logs is not None:
                logs.append(
                    f"[CLARIFY] Ambiguous value '{term_l}' found in columns: {', '.join(matches[:6])}"
                )
            return _build_custom_chart_clarification(term_l, matches, reason="value_match")

    return None

def _to_month_label_if_possible(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Already a date-like object
    try:
        ts = pd.Timestamp(value)
        if not pd.isna(ts):
            # Avoid accidental conversion of plain numerics by checking original form.
            raw = str(value).strip()
            if '-' in raw or '/' in raw or len(raw) in (6, 8) or hasattr(value, 'year'):
                return ts.strftime('%b %Y')
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return None

    # Normalize common numeric forms like 202301.0
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split('.', 1)[0]

    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) < 6:
        return None

    year = int(digits[:4])
    month = int(digits[4:6])
    if 1900 <= year <= 2100 and 1 <= month <= 12:
        try:
            return pd.Timestamp(year=year, month=month, day=1).strftime('%b %Y')
        except Exception:
            return None

    return None


def _normalize_month_axis_labels(values, title='', xlabel=''):
    if not values:
        return values

    hint = f"{title} {xlabel}".lower()
    force_month = ('month' in hint) or ('monthly' in hint) or ('yyyy_mm' in hint)

    converted = []
    converted_count = 0
    for v in values:
        month_label = _to_month_label_if_possible(v)
        if month_label is not None:
            converted.append(month_label)
            converted_count += 1
        else:
            converted.append(v)

    if force_month or converted_count >= max(2, len(values) // 2):
        return [str(v) for v in converted]

    return values


def _aggregate_xy_by_x(x_values, y_values):
    if not x_values or not y_values or len(x_values) != len(y_values):
        return x_values, y_values

    ordered_keys = []
    sums = {}
    for x, y in zip(x_values, y_values):
        key = str(x)
        if key not in sums:
            sums[key] = 0.0
            ordered_keys.append(key)
        try:
            sums[key] += float(y)
        except Exception:
            pass

    return ordered_keys, [sums[k] for k in ordered_keys]


def _normalize_and_aggregate_line_series(x_values, y_values, title='', xlabel=''):
    normalized_x = _normalize_month_axis_labels(x_values, title=title, xlabel=xlabel)
    if normalized_x is None:
        return x_values, y_values

    if normalized_x != x_values:
        return _aggregate_xy_by_x(normalized_x, y_values)

    return normalized_x, y_values




def _extract_kpi_sparkline_from_df(df, target_points=7):
    if df is None or df.empty or df.shape[1] < 2:
        return []

    work = pd.DataFrame({
        "x": df.iloc[:, 0],
        "y": pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
    })

    # Prefer chronological sorting when x looks like date/timestamp.
    parsed_ts = pd.to_datetime(work["x"], errors="coerce")
    if parsed_ts.notna().sum() >= max(2, len(work) // 2):
        # Normalize to monthly buckets to reduce noisy day-level jumps in KPI cards.
        work = work.assign(_ts=parsed_ts).dropna(subset=["_ts"])
        work = (
            work.assign(_period=work["_ts"].dt.to_period("M").dt.to_timestamp())
            .groupby("_period", as_index=False)["y"]
            .sum()
            .sort_values("_period", kind="mergesort")
        )

        # Drop current month point when present; it is often incomplete and can flip trend sign.
        if not work.empty:
            current_month_start = pd.Timestamp.now().to_period("M").to_timestamp()
            if work.iloc[-1]["_period"] == current_month_start and len(work) >= 2:
                work = work.iloc[:-1]
    else:
        # Fallback to numeric sorting when possible.
        parsed_num = pd.to_numeric(work["x"], errors="coerce")
        if parsed_num.notna().sum() >= max(2, len(work) // 2):
            work = work.assign(_k=parsed_num).sort_values("_k", kind="mergesort")

    values = [float(v) for v in work["y"].tolist()]
    if not values:
        return []

    if len(values) >= target_points:
        values = values[-target_points:]
    else:
        # Left-pad with oldest value to keep sequence shape deterministic.
        values = [values[0]] * (target_points - len(values)) + values

    # Guard against edge outliers (first/last partial-period artifacts).
    if len(values) >= 4:
        core = values[1:]
        median_core = float(np.median(core))
        if median_core > 0 and values[0] < 0.35 * median_core:
            values[0] = values[1]
        if median_core > 0 and values[-1] < 0.35 * median_core:
            values[-1] = values[-2]
        if median_core > 0 and values[0] > 2.8 * median_core:
            values[0] = values[1]
        if median_core > 0 and values[-1] > 2.8 * median_core:
            values[-1] = values[-2]

    return values


def _resolve_table_column_labels(columns, xlabel="", ylabel=""):
    names = [str(c) for c in (columns or [])]
    if not names:
        return []

    generic = {"columns", "rows", "column", "row", "label", "labels", "value", "values", "x", "y"}
    out = []
    for idx, name in enumerate(names):
        n = str(name).strip()
        lower = n.lower()
        if lower in generic:
            if idx == 0:
                out.append(str(xlabel).strip() or "Label")
            elif idx == 1:
                out.append(str(ylabel).strip() or "Value")
            else:
                out.append(f"Field {idx+1}")
        else:
            out.append(n)
    return out

def _build_custom_chart_payload(plan, df):
    chart_type = str(plan.get("type", "bar")).lower()
    if chart_type not in ALLOWED_CUSTOM_CHART_TYPES:
        chart_type = "bar"

    chart_data = {
        "id": "custom_generated",
        "title": plan.get("title", "Custom Chart"),
        "type": chart_type,
        "xlabel": plan.get("xlabel", ""),
        "ylabel": plan.get("ylabel", ""),
        "x": [],
        "y": [],
        "z": [],
        "columns": [],
        "rows": [],
        "labels": [],
        "parents": [],
        "values": [],
        "measure": []
    }

    if df is None or df.empty:
        chart_data["title"] = f"{chart_data['title']} (No Data)"
        return chart_data

    df = df.copy()
    raw_columns = [str(c) for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]

    if chart_type == "table":
        chart_data["columns"] = _resolve_table_column_labels(raw_columns, chart_data.get("xlabel", ""), chart_data.get("ylabel", ""))
        chart_data["rows"] = df.fillna("").astype(str).values.tolist()
    elif chart_type == "treemap":
        if "labels" in df and "values" in df:
            chart_data["labels"] = df["labels"].astype(str).tolist()
            chart_data["parents"] = df["parents"].astype(str).tolist() if "parents" in df else [""] * len(chart_data["labels"])
            chart_data["values"] = df["values"].fillna(0).astype(float).tolist()
        elif len(df.columns) >= 2:
            chart_data["labels"] = df.iloc[:, 0].astype(str).tolist()
            chart_data["parents"] = [""] * len(chart_data["labels"])
            chart_data["values"] = df.iloc[:, 1].fillna(0).astype(float).tolist()
    elif chart_type == "waterfall":
        if len(df.columns) >= 2:
            chart_data["x"] = df["x"].astype(str).tolist() if "x" in df else df.iloc[:, 0].astype(str).tolist()
            chart_data["y"] = df["y"].fillna(0).astype(float).tolist() if "y" in df else df.iloc[:, 1].fillna(0).astype(float).tolist()
            chart_data["measure"] = df["measure"].astype(str).tolist() if "measure" in df else ["relative"] * len(chart_data["x"])
    elif chart_type == "heatmap":
        if len(df.columns) >= 3:
            chart_data["x"] = df["x"].astype(str).tolist() if "x" in df else df.iloc[:, 0].astype(str).tolist()
            chart_data["y"] = df["y"].astype(str).tolist() if "y" in df else df.iloc[:, 1].astype(str).tolist()
            chart_data["z"] = df["z"].fillna(0).astype(float).tolist() if "z" in df else df.iloc[:, 2].fillna(0).astype(float).tolist()
    else:
        if len(df.columns) >= 2:
            chart_data["x"] = df["x"].tolist() if "x" in df else df.iloc[:, 0].tolist()
            chart_data["y"] = df["y"].tolist() if "y" in df else df.iloc[:, 1].tolist()
            if chart_type == "line":
                chart_data["x"], chart_data["y"] = _normalize_and_aggregate_line_series(
                    chart_data["x"], chart_data["y"], chart_data.get("title", ""), chart_data.get("xlabel", "")
                )
            else:
                chart_data["x"] = _normalize_month_axis_labels(chart_data["x"], chart_data.get("title", ""), chart_data.get("xlabel", ""))

    return chart_data

def generate_custom_kpi_plan(final_schema_context, user_prompt, debug_logs=None, table_name="final_view"):
    prompt = f"""
    You are a senior BI analyst. Build one KPI spec from a natural language request.

    DATA SCHEMA (DuckDB table: final_view):
    {final_schema_context}

    USER REQUEST:
    {user_prompt}

    RULES:
    1. Return ONLY JSON.
    2. SQL must be SELECT/WITH only and must use only final_view.
    3. Provide two queries:
       - value_sql: one-row KPI value query (single metric cell preferred).
       - trend_sql: time trend query with aliases exactly x, y for KPI sparkline.
    4. For trend_sql use a date/timestamp column when available and order by x.
    5. final_view is transaction-level. For transactional metrics aggregate directly.
    6. For master/entity attributes (rating, age, salary, static price, scores, etc.), deduplicate by entity key before AVG-style aggregations.
    7. For unique entity questions use COUNT(DISTINCT <detected_entity_key>) instead of COUNT(*).

    JSON format:
    {{
      "label": "Total Revenue",
      "value_sql": "SELECT SUM(revenue) AS value FROM final_view",
      "trend_sql": "SELECT CAST(order_date AS DATE) AS x, SUM(revenue) AS y FROM final_view GROUP BY 1 ORDER BY 1 LIMIT 24"
    }}
    """
    if table_name != "final_view":
        prompt = prompt.replace("final_view", table_name)

    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=True, debug_logs=debug_logs, context="Generate Custom KPI Plan")
    try:
        parsed = json.loads(res) if res else None
    except Exception:
        parsed = None
    return parsed, tokens


def _default_custom_kpi_plan_from_columns(schema_columns, user_prompt, table_name="final_view"):
    text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
    num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
    date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]

    prompt_l = str(user_prompt or "").lower()
    wants_count = any(tok in prompt_l for tok in ["count", "number of", "how many", "records", "invoices"])

    if wants_count or not num_cols:
        label = "Record Count"
        value_sql = f"SELECT COUNT(*) AS value FROM {table_name}"
        if date_cols:
            d_col = _quote_identifier(date_cols[0])
            trend_sql = f"SELECT CAST({d_col} AS DATE) AS x, COUNT(*) AS y FROM {table_name} GROUP BY 1 ORDER BY 1 LIMIT 24"
        else:
            trend_sql = ""
        return {
            "label": label,
            "value_sql": value_sql,
            "trend_sql": trend_sql,
        }

    metric_col = _quote_identifier(num_cols[0])
    metric_name = num_cols[0].replace("_", " ").title()
    label = f"Total {metric_name}"
    value_sql = f"SELECT SUM({metric_col}) AS value FROM {table_name}"

    trend_sql = ""
    if date_cols:
        d_col = _quote_identifier(date_cols[0])
        trend_sql = (
            f"SELECT CAST({d_col} AS DATE) AS x, SUM({metric_col}) AS y "
            f"FROM {table_name} GROUP BY 1 ORDER BY 1 LIMIT 24"
        )

    return {
        "label": label,
        "value_sql": value_sql,
        "trend_sql": trend_sql,
    }


def _extract_first_scalar(df, default=0.0):
    if df is None or df.empty or df.shape[1] < 1:
        return default

    raw_val = df.iloc[0, 0]
    try:
        if pd.isna(raw_val):
            return default
    except Exception:
        pass

    return raw_val


def _format_kpi_display_value(value):
    if value is None:
        return "0"

    try:
        if pd.isna(value):
            return "0"
    except Exception:
        pass

    try:
        n = float(value)
        if abs(n) >= 100 or float(n).is_integer():
            return f"{n:,.0f}"
        return f"{n:,.2f}"
    except Exception:
        return str(value)


def _build_custom_kpi_payload(plan, value_raw, sparkline_data):
    label = str((plan or {}).get("label") or "Custom KPI").strip() or "Custom KPI"
    if not sparkline_data:
        base = 0.0
        try:
            base = float(value_raw)
        except Exception:
            base = 0.0
        sparkline_data = [base] * 7

    return {
        "label": label,
        "value": _format_kpi_display_value(value_raw),
        "sparkline": [float(v) for v in sparkline_data],
    }


def generate_custom_kpi_from_prompt_databricks(user_prompt, active_filters_json='{}'):
    connection = get_databricks_connection()
    llm_logs = []
    try:
        include_sample_rows = _llm_include_sample_rows()
        if not include_sample_rows:
            llm_logs.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=llm_logs,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = date_cols[0] if date_cols else None

        where_sql, filter_count = _build_databricks_where_clause(
            active_filters_json,
            column_names,
            date_column=date_column,
        )
        if filter_count > 0:
            llm_logs.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        ai_plan, tokens_used = generate_custom_kpi_plan(
            schema_context,
            user_prompt,
            debug_logs=llm_logs,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        if not ai_plan:
            ai_plan = _default_custom_kpi_plan_from_columns(
                schema_columns,
                user_prompt,
                table_name=DATABRICKS_LOGICAL_VIEW_NAME,
            )

        value_sql = str(ai_plan.get("value_sql") or ai_plan.get("sql") or "").strip()
        value_sql = value_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
        value_sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(value_sql)
        for note in guardrail_notes:
            llm_logs.append(f"[SECURITY] {note}")

        if not _safe_custom_sql(value_sql):
            ai_plan = _default_custom_kpi_plan_from_columns(
                schema_columns,
                user_prompt,
                table_name=DATABRICKS_LOGICAL_VIEW_NAME,
            )
            value_sql = str(ai_plan.get("value_sql") or "").strip()

        value_df, executed_value_sql = _execute_databricks_user_sql(
            connection,
            value_sql,
            source_table_base,
            query_source=source_table_query,
            where_sql=where_sql,
            logs=llm_logs,
            context="Custom KPI Value",
        )
        value_raw = _extract_first_scalar(value_df, default=0.0)

        trend_sql = str(ai_plan.get("trend_sql") or "").strip()
        trend_sql = trend_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
        executed_trend_sql = ""
        sparkline_data = []

        if trend_sql:
            trend_sql, trend_notes = _apply_sql_security_and_cost_guardrails(trend_sql)
            for note in trend_notes:
                llm_logs.append(f"[SECURITY] {note}")

            if _safe_custom_sql(trend_sql):
                try:
                    trend_df, executed_trend_sql = _execute_databricks_user_sql(
                        connection,
                        trend_sql,
                        source_table_base,
                        query_source=source_table_query,
                        where_sql=where_sql,
                        logs=llm_logs,
                        context="Custom KPI Trend",
                    )
                    if not trend_df.empty:
                        sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=7)
                except Exception as e:
                    llm_logs.append(f"[WARN] KPI trend query failed: {str(e)}")

        if not sparkline_data:
            try:
                base = float(value_raw)
            except Exception:
                base = 0.0
            sparkline_data = [base] * 7

        kpi_payload = _build_custom_kpi_payload(ai_plan, value_raw, sparkline_data)
        return {
            "kpi": kpi_payload,
            "generated_sql": {
                "value_sql": executed_value_sql,
                "trend_sql": executed_trend_sql,
            },
            "tokens_used": tokens_used,
            "logs": llm_logs,
            "data_mode": "databricks",
        }
    finally:
        connection.close()


def execute_dashboard_logic_databricks(active_filters_json=None, session_id=None):
    log = []
    total_tokens = 0
    if not session_id:
        session_id = str(uuid.uuid4())

    include_sample_rows = _llm_include_sample_rows()
    if not include_sample_rows:
        log.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

    connection = get_databricks_connection()
    try:
        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=log,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]

        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
        num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
        date_column = date_cols[0] if date_cols else None

        where_sql, filter_count = _build_databricks_where_clause(
            active_filters_json,
            column_names,
            date_column=date_column,
        )
        if filter_count > 0:
            log.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        date_range = {"min": None, "max": None}
        if date_column:
            date_ident = _quote_identifier(date_column)
            try:
                range_df = fetch_dataframe(
                    connection,
                    f"SELECT MIN(CAST({date_ident} AS DATE)) AS min_date, MAX(CAST({date_ident} AS DATE)) AS max_date FROM {source_table_query}",
                    readonly=True,
                )
                if not range_df.empty:
                    min_val = range_df.iloc[0].get("min_date")
                    max_val = range_df.iloc[0].get("max_date")
                    if pd.notna(min_val):
                        date_range["min"] = str(min_val)
                    if pd.notna(max_val):
                        date_range["max"] = str(max_val)
            except Exception as e:
                log.append(f"[WARN] Could not compute date range: {str(e)}")

        col_text = " ".join(column_names).lower()
        forced_domain = None
        if any(x in col_text for x in ['shipment', 'supplier', 'procurement']):
            forced_domain = "Supply Chain"
        elif any(x in col_text for x in ['campaign', 'ad_spend', 'click']):
            forced_domain = "Marketing"
        elif any(x in col_text for x in ['employee', 'salary', 'payroll']):
            forced_domain = "Human Resources"

        if _is_databricks_mode_active() and STRICT_SQL_GUARDRAILS:
            log.append(
                f"[SECURITY] Databricks SQL guardrails active (SELECT/WITH only, forbidden keywords blocked, max LIMIT={AI_SQL_MAX_LIMIT})"
            )

        plan, tokens = generate_viz_config(
            schema_context,
            forced_domain,
            debug_logs=log,
            logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        total_tokens += tokens
        if not plan:
            metric_col = num_cols[0] if num_cols else None
            dim_col = text_cols[0] if text_cols else None
            time_col = date_cols[0] if date_cols else None
            fallback_sql = (
                f"SELECT CAST({_quote_identifier(time_col)} AS DATE) AS x, SUM({_quote_identifier(metric_col)}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1"
                if time_col and metric_col
                else f"SELECT 'All Data' AS x, COUNT(*) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME}"
            )
            plan = {
                "domain": forced_domain or "General",
                "filters": text_cols[:3],
                "kpis": [
                    {"label": "Record Count", "sql": f"SELECT COUNT(*) FROM {DATABRICKS_LOGICAL_VIEW_NAME}"},
                ],
                "charts": [
                    {"title": "Overview", "type": "line" if time_col else "bar", "sql": fallback_sql, "xlabel": "", "ylabel": ""}
                ],
            }
        domain = plan.get("domain", "General")
        if domain not in SUPPORTED_DOMAINS:
            domain = forced_domain or "General"
        theme = dict(FIXED_DASHBOARD_THEME)

        output = {
            "domain": domain,
            "theme": theme,
            "filters": [],
            "kpis": [],
            "charts": [],
            "logs": log,
            "date_range": date_range,
            "has_date_column": date_column is not None,
            "tokens_used": total_tokens,
            "master_preview": None,
            "data_mode": "databricks",
            "source_table": source_table_base,
            "session_id": session_id,
        }

        if include_sample_rows:
            try:
                preview_df = fetch_dataframe(connection, f"SELECT * FROM {source_table_query} LIMIT 3", readonly=True)
                output["master_preview"] = {
                    "columns": preview_df.columns.tolist(),
                    "rows": preview_df.values.tolist(),
                }
            except Exception as e:
                log.append(f"[WARN] Could not load Databricks preview rows: {str(e)}")

        filter_candidates = []
        seen = set()
        for col in (plan.get("filters", []) + text_cols):
            c = str(col)
            if c.lower() in seen:
                continue
            seen.add(c.lower())
            filter_candidates.append(c)

        for col in filter_candidates:
            try:
                col_ident = _quote_identifier(col)
                clauses = []
                if where_sql:
                    clauses.append(where_sql)
                clauses.append(f"{col_ident} IS NOT NULL")

                sample_rows = _safe_int_env("DATABRICKS_FILTER_VALUE_SAMPLE_ROWS", 100000)
                base_q = f"SELECT {col_ident} AS raw_v FROM {source_table_query}"
                if clauses:
                    base_q += " WHERE " + " AND ".join(clauses)
                if sample_rows > 0:
                    base_q += f" LIMIT {sample_rows}"

                q = f"SELECT DISTINCT CAST(raw_v AS STRING) AS v FROM ({base_q}) __fv LIMIT 50"
                vals_df = fetch_dataframe(connection, q, readonly=True)
                vals = [str(v) for v in vals_df["v"].dropna().tolist()] if "v" in vals_df else []
                if vals:
                    output["filters"].append({
                        "label": col.replace('_', ' ').title(),
                        "column": col,
                        "values": vals,
                    })
            except Exception:
                continue

        def _run_kpi_spec(label, kpi_sql, trend_sql=None, context_prefix="KPI"):
            label_text = str(label or "Metric").strip() or "Metric"
            kpi_sql = str(kpi_sql or "").replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
            if not kpi_sql:
                raise ValueError("Empty KPI SQL")

            df, _ = _execute_databricks_user_sql(
                connection,
                kpi_sql,
                source_table_base,
                query_source=source_table_query,
                where_sql=where_sql,
                logs=log,
                context=f"{context_prefix} {label_text}",
            )

            val = 0
            if not df.empty and df.shape[1] >= 1:
                raw_val = df.iloc[0, 0]
                val = 0 if pd.isna(raw_val) else raw_val

            sparkline_data = []
            if trend_sql:
                trend_sql = str(trend_sql).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
                if trend_sql:
                    try:
                        trend_df, _ = _execute_databricks_user_sql(
                            connection,
                            trend_sql,
                            source_table_base,
                            query_source=source_table_query,
                            where_sql=where_sql,
                            logs=log,
                            context=f"{context_prefix} Trend {label_text}",
                        )
                        if not trend_df.empty and trend_df.shape[1] >= 2:
                            sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=7)
                    except Exception as e:
                        log.append(f"[WARN] KPI trend query failed for {label_text}: {str(e)}")

            if not sparkline_data:
                try:
                    base_val = float(val)
                except Exception:
                    base_val = 0.0
                sparkline_data = [max(0.0, base_val)] * 7

            return {
                "label": label_text,
                "value": _format_kpi_display_value(val),
                "sparkline": [float(v) for v in sparkline_data],
            }

        filtered_row_count = 0
        try:
            count_df = fetch_dataframe(
                connection,
                f"SELECT COUNT(*) AS c FROM {source_table_query}" + (f" WHERE {where_sql}" if where_sql else ""),
                readonly=True,
            )
            if not count_df.empty:
                filtered_row_count = int(count_df.iloc[0].get("c") or 0)
        except Exception as e:
            log.append(f"[WARN] Could not compute filtered row count for KPI fallback: {str(e)}")

        kpis = plan.get("kpis", [])
        valid_kpis = []
        used_labels = set()

        for kpi in kpis:
            label = str(kpi.get("label", "Metric")).strip() or "Metric"
            label_key = label.lower()
            if label_key in used_labels:
                continue
            try:
                valid_kpis.append(
                    _run_kpi_spec(
                        label=label,
                        kpi_sql=kpi.get("sql", ""),
                        trend_sql=kpi.get("trend_sql"),
                        context_prefix="KPI",
                    )
                )
                used_labels.add(label_key)
            except Exception as e:
                log.append(f"[WARN] KPI failed ({label}): {str(e)}")

        preferred_numeric_names = [
            "netamount",
            "gross_value",
            "total_value",
            "amount",
            "revenue",
            "sales",
            "invoicequantity",
            "quantity",
        ]
        fallback_metric_col = None
        if num_cols:
            fallback_metric_col = num_cols[0]
            for candidate in preferred_numeric_names:
                matched = next((c for c in num_cols if str(c).lower() == candidate), None)
                if matched:
                    fallback_metric_col = matched
                    break

        entity_count_col = next((c for c in column_names if str(c).lower() in {"invoicenumber", "billdetailedrid", "inv_item_guid", "retaileruid"}), None)
        fallback_kpis = [
            {
                "label": "Record Count",
                "sql": f"SELECT COUNT(*) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, COUNT(*) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
            },
            {
                "label": "Distinct Invoices",
                "sql": f"SELECT COUNT(DISTINCT {_quote_identifier(entity_count_col)}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, COUNT(DISTINCT {_quote_identifier(entity_count_col)}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
            } if entity_count_col else None,
            {
                "label": f"Total {fallback_metric_col.replace('_', ' ').title()}",
                "sql": f"SELECT SUM({_quote_identifier(fallback_metric_col)}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, SUM({_quote_identifier(fallback_metric_col)}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if (date_column and fallback_metric_col) else "",
            } if fallback_metric_col else None,
            {
                "label": f"Avg {fallback_metric_col.replace('_', ' ').title()}",
                "sql": f"SELECT AVG({_quote_identifier(fallback_metric_col)}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, AVG({_quote_identifier(fallback_metric_col)}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if (date_column and fallback_metric_col) else "",
            } if fallback_metric_col else None,
        ]

        if filtered_row_count > 0:
            for ncol in num_cols:
                if not ncol:
                    continue
                ncol_key = str(ncol).lower()
                if fallback_metric_col and ncol_key == str(fallback_metric_col).lower():
                    continue
                ncol_ident = _quote_identifier(ncol)
                ncol_name = str(ncol).replace('_', ' ').title()
                fallback_kpis.append({
                    "label": f"Total {ncol_name}",
                    "sql": f"SELECT SUM({ncol_ident}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                    "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, SUM({ncol_ident}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
                })
                fallback_kpis.append({
                    "label": f"Avg {ncol_name}",
                    "sql": f"SELECT AVG({ncol_ident}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                    "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, AVG({ncol_ident}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
                })
                if len(fallback_kpis) >= 16:
                    break

            for tcol in text_cols:
                if not tcol:
                    continue
                tcol_ident = _quote_identifier(tcol)
                tcol_name = str(tcol).replace('_', ' ').title()
                fallback_kpis.append({
                    "label": f"Distinct {tcol_name}",
                    "sql": f"SELECT COUNT(DISTINCT {tcol_ident}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                    "trend_sql": "",
                })
                if len(fallback_kpis) >= 24:
                    break

        if filtered_row_count > 0 and len(valid_kpis) < 4:
            for fallback_kpi in fallback_kpis:
                if not fallback_kpi or len(valid_kpis) >= 4:
                    continue
                label = str(fallback_kpi["label"]).strip() or "Metric"
                label_key = label.lower()
                if label_key in used_labels:
                    continue
                try:
                    valid_kpis.append(
                        _run_kpi_spec(
                            label=label,
                            kpi_sql=fallback_kpi["sql"],
                            trend_sql=fallback_kpi.get("trend_sql"),
                            context_prefix="KPI Fallback",
                        )
                    )
                    used_labels.add(label_key)
                except Exception as e:
                    log.append(f"[WARN] KPI fallback failed ({label}): {str(e)}")

        while len(valid_kpis) < 4:
            if filtered_row_count <= 0:
                valid_kpis.append({"label": "No Data", "value": "-", "sparkline": [0.0] * 7})
            else:
                valid_kpis.append({
                    "label": f"Rows (Filtered) {len(valid_kpis)+1}",
                    "value": _format_kpi_display_value(filtered_row_count),
                    "sparkline": [float(filtered_row_count)] * 7,
                })

        output["kpis"] = valid_kpis[:4]

        charts = plan.get("charts", [])[:5]

        fallback_dims = []
        seen_fallback_dims = set()
        for col in (text_cols + [f.get("column") for f in output.get("filters", []) if isinstance(f, dict)]):
            c = str(col or "").strip()
            if not c:
                continue
            key = c.lower()
            if key in seen_fallback_dims:
                continue
            seen_fallback_dims.add(key)
            fallback_dims.append(c)

        fallback_metric = num_cols[0] if num_cols else None
        fallback_metric_ident = _quote_identifier(fallback_metric) if fallback_metric else None

        def _apply_databricks_chart_fallback(c_data, chart_index, reason_label):
            for dim_col in fallback_dims:
                dim_ident = _quote_identifier(dim_col)

                if fallback_metric_ident:
                    fallback_sql = (
                        f"SELECT CAST({dim_ident} AS STRING) AS x, SUM({fallback_metric_ident}) AS y "
                        f"FROM {DATABRICKS_LOGICAL_VIEW_NAME} "
                        f"WHERE {dim_ident} IS NOT NULL "
                        f"GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
                    )
                    fallback_title = (
                        f"Top {dim_col.replace('_', ' ').title()} by "
                        f"{fallback_metric.replace('_', ' ').title()}"
                    )
                    fallback_ylabel = fallback_metric.replace('_', ' ').title()
                else:
                    fallback_sql = (
                        f"SELECT CAST({dim_ident} AS STRING) AS x, COUNT(*) AS y "
                        f"FROM {DATABRICKS_LOGICAL_VIEW_NAME} "
                        f"WHERE {dim_ident} IS NOT NULL "
                        f"GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
                    )
                    fallback_title = f"Record Count by {dim_col.replace('_', ' ').title()}"
                    fallback_ylabel = "Count"

                try:
                    fallback_df, _ = _execute_databricks_user_sql(
                        connection,
                        fallback_sql,
                        source_table_base,
                        query_source=source_table_query,
                        where_sql=where_sql,
                        logs=log,
                        context=f"Chart {chart_index} Fallback",
                    )
                    if fallback_df.empty or fallback_df.shape[1] < 2:
                        continue

                    fallback_df.columns = [str(c).lower() for c in fallback_df.columns]
                    c_data["type"] = "bar"
                    c_data["x"] = fallback_df["x"].astype(str).tolist() if "x" in fallback_df else fallback_df.iloc[:, 0].astype(str).tolist()
                    c_data["y"] = fallback_df["y"].fillna(0).astype(float).tolist() if "y" in fallback_df else fallback_df.iloc[:, 1].fillna(0).astype(float).tolist()
                    c_data["x"] = _normalize_month_axis_labels(c_data["x"], c_data.get("title", ""), c_data.get("xlabel", ""))
                    c_data["z"] = []
                    c_data["title"] = fallback_title
                    c_data["xlabel"] = dim_col.replace('_', ' ').title()
                    c_data["ylabel"] = fallback_ylabel
                    log.append(f"[INFO] Chart {chart_index} fallback applied ({reason_label}) using {dim_col}")
                    return True
                except Exception as fallback_err:
                    log.append(f"[WARN] Chart {chart_index} fallback failed for {dim_col}: {str(fallback_err)}")
                    continue

            c_data["title"] = f"{c_data['title']} ({reason_label})"
            return False

        for i, chart in enumerate(charts):
            chart_type = str(chart.get("type", "bar")).lower()
            if chart_type not in ALLOWED_CUSTOM_CHART_TYPES:
                chart_type = "bar"

            c_data = {
                "id": f"chart_{i}",
                "title": chart.get("title", f"Chart {i+1}"),
                "type": chart_type,
                "xlabel": chart.get("xlabel", ""),
                "ylabel": chart.get("ylabel", ""),
                "x": [],
                "y": [],
                "z": [],
                "columns": [],
                "rows": [],
                "labels": [],
                "parents": [],
                "values": [],
                "measure": [],
            }

            chart_has_data = False
            failure_reason = None

            try:
                chart_sql = str(chart.get("sql", "")).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
                df, _ = _execute_databricks_user_sql(
                    connection,
                    chart_sql,
                    source_table_base,
                    query_source=source_table_query,
                    where_sql=where_sql,
                    logs=log,
                    context=f"Chart {i}",
                )

                if not df.empty:
                    df.columns = [str(c).lower() for c in df.columns]
                    if chart_type == "table":
                        c_data["columns"] = _resolve_table_column_labels(df.columns, c_data.get("xlabel", ""), c_data.get("ylabel", ""))
                        c_data["rows"] = df.fillna("").astype(str).values.tolist()
                        chart_has_data = bool(c_data["rows"])
                    elif chart_type == "treemap":
                        if "labels" in df and "values" in df:
                            c_data["labels"] = df["labels"].astype(str).tolist()
                            c_data["parents"] = df["parents"].astype(str).tolist() if "parents" in df else [""] * len(c_data["labels"])
                            c_data["values"] = df["values"].fillna(0).astype(float).tolist()
                        elif len(df.columns) >= 2:
                            c_data["labels"] = df.iloc[:, 0].astype(str).tolist()
                            c_data["parents"] = [""] * len(c_data["labels"])
                            c_data["values"] = df.iloc[:, 1].fillna(0).astype(float).tolist()
                        chart_has_data = bool(c_data["labels"]) and bool(c_data["values"])
                    elif chart_type == "waterfall":
                        if len(df.columns) >= 2:
                            c_data["x"] = df["x"].astype(str).tolist() if "x" in df else df.iloc[:, 0].astype(str).tolist()
                            c_data["y"] = df["y"].fillna(0).astype(float).tolist() if "y" in df else df.iloc[:, 1].fillna(0).astype(float).tolist()
                            c_data["measure"] = df["measure"].astype(str).tolist() if "measure" in df else ["relative"] * len(c_data["x"])
                            chart_has_data = bool(c_data["x"]) and bool(c_data["y"])
                    elif chart_type == "heatmap":
                        if len(df.columns) >= 3:
                            c_data["x"] = df["x"].astype(str).tolist() if "x" in df else df.iloc[:, 0].astype(str).tolist()
                            c_data["y"] = df["y"].astype(str).tolist() if "y" in df else df.iloc[:, 1].astype(str).tolist()
                            c_data["z"] = df["z"].fillna(0).astype(float).tolist() if "z" in df else df.iloc[:, 2].fillna(0).astype(float).tolist()
                            chart_has_data = bool(c_data["x"]) and bool(c_data["y"]) and bool(c_data["z"])
                    else:
                        if len(df.columns) >= 2:
                            c_data["x"] = df["x"].tolist() if "x" in df else df.iloc[:, 0].tolist()
                            c_data["y"] = df["y"].tolist() if "y" in df else df.iloc[:, 1].tolist()
                            if chart_type == "line":
                                c_data["x"], c_data["y"] = _normalize_and_aggregate_line_series(
                                    c_data["x"], c_data["y"], c_data.get("title", ""), c_data.get("xlabel", "")
                                )
                            else:
                                c_data["x"] = _normalize_month_axis_labels(c_data["x"], c_data.get("title", ""), c_data.get("xlabel", ""))
                            chart_has_data = bool(c_data["x"]) and bool(c_data["y"])
                else:
                    failure_reason = "No Data"

                if not chart_has_data and failure_reason is None:
                    failure_reason = "No Data"
            except Exception as e:
                log.append(f"[WARN] Chart {i} failed: {str(e)}")
                failure_reason = "Error"

            if failure_reason:
                _apply_databricks_chart_fallback(c_data, i, failure_reason)

            output["charts"].append(c_data)

        while len(output["charts"]) < 5:
            idx = len(output["charts"])
            output["charts"].append({
                "id": f"chart_{idx}",
                "title": f"Chart {idx+1} (No Data)",
                "type": "bar",
                "xlabel": "",
                "ylabel": "",
                "x": [],
                "y": [],
                "z": [],
                "columns": [],
                "rows": [],
                "labels": [],
                "parents": [],
                "values": [],
                "measure": [],
            })

        return output
    finally:
        connection.close()


def generate_custom_chart_from_prompt_databricks(user_prompt, active_filters_json='{}', clarification_choice=None):
    connection = get_databricks_connection()
    llm_logs = []
    try:
        include_sample_rows = _llm_include_sample_rows()
        if not include_sample_rows:
            llm_logs.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=llm_logs,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = date_cols[0] if date_cols else None

        where_sql, filter_count = _build_databricks_where_clause(
            active_filters_json,
            column_names,
            date_column=date_column,
        )
        if filter_count > 0:
            llm_logs.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        normalized_choice = _normalize_custom_chart_clarification_choice(clarification_choice, schema_columns)
        ambiguity = _detect_custom_chart_ambiguity(
            connection,
            user_prompt,
            schema_columns,
            source_table_query,
            where_sql=where_sql,
            clarification_choice=normalized_choice,
            logs=llm_logs,
        )
        if ambiguity:
            return {
                "needs_clarification": True,
                "clarification": ambiguity,
                "tokens_used": 0,
                "logs": llm_logs,
                "data_mode": "databricks",
            }

        prompt_for_chart = _apply_custom_chart_clarification_to_prompt(user_prompt, normalized_choice, logs=llm_logs)

        ai_plan, tokens_used = generate_custom_chart_plan(
            schema_context,
            prompt_for_chart,
            debug_logs=llm_logs,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        if not ai_plan:
            ai_plan = _default_custom_chart_plan_from_columns(
            schema_columns,
            user_prompt,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )

        sql = str(ai_plan.get("sql", "")).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
        sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(sql)
        for note in guardrail_notes:
            llm_logs.append(f"[SECURITY] {note}")

        if not _safe_custom_sql(sql):
            ai_plan = _default_custom_chart_plan_from_columns(
            schema_columns,
            user_prompt,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
            sql = ai_plan["sql"]
            sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(sql)
            for note in guardrail_notes:
                llm_logs.append(f"[SECURITY] {note}")

        df, executed_sql = _execute_databricks_user_sql(
            connection,
            sql,
            source_table_base,
            query_source=source_table_query,
            where_sql=where_sql,
            logs=llm_logs,
            context="Custom Chart",
        )

        chart_payload = _build_custom_chart_payload(ai_plan, df)
        return {
            "chart": chart_payload,
            "generated_sql": executed_sql,
            "tokens_used": tokens_used,
            "logs": llm_logs,
            "data_mode": "databricks",
        }
    finally:
        connection.close()


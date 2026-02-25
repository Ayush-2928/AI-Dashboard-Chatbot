import pandas as pd
import json
import os
import re
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
KPI_TREND_POINTS = 6


def _is_databricks_mode_active():
    return config_is_databricks_mode_active()


def _llm_include_sample_rows():
    return config_llm_include_sample_rows()


def _default_dashboard_months():
    raw = os.getenv("DASHBOARD_DEFAULT_MONTHS", "12")
    try:
        return max(1, int(raw))
    except Exception:
        return 12


def _compute_default_date_window(max_date_hint=None, months=None):
    n_months = months or _default_dashboard_months()
    end_ts = pd.to_datetime(max_date_hint, errors="coerce") if max_date_hint else pd.NaT
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.utcnow().normalize()
    else:
        end_ts = pd.Timestamp(end_ts).normalize()

    start_ts = (end_ts - pd.DateOffset(months=n_months)) + pd.Timedelta(days=1)
    return str(start_ts.date()), str(end_ts.date())


def _apply_default_date_filters(active_filters_json, date_column, date_range_override=None, logs=None):
    filters = _parse_active_filters_json(active_filters_json)
    if not date_column:
        return json.dumps(filters), "", ""

    start_raw = str(filters.get("_start_date") or "").strip()
    end_raw = str(filters.get("_end_date") or "").strip()

    # Respect explicit user-provided date bounds.
    if start_raw or end_raw:
        if not filters.get("_date_column"):
            filters["_date_column"] = date_column
        return json.dumps(filters), start_raw, end_raw

    max_hint = None
    if isinstance(date_range_override, dict):
        max_hint = date_range_override.get("max")

    start_d, end_d = _compute_default_date_window(max_date_hint=max_hint)
    filters["_date_column"] = date_column
    filters["_start_date"] = start_d
    filters["_end_date"] = end_d

    if logs is not None:
        logs.append(f"[FILTER] Default date window applied: {start_d} to {end_d} (last {_default_dashboard_months()} months)")

    return json.dumps(filters), start_d, end_d


def _redact_sensitive_text(text):
    return guardrail_redact_sensitive_text(text)

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



def _select_invoice_entity_key(schema_columns):
    best_col = None
    best_score = -10_000

    for col_name, col_type in (schema_columns or []):
        name = str(col_name or "").strip()
        if not name:
            continue

        n = name.lower()
        has_invoice = "invoice" in n
        has_bill = "bill" in n
        if not (has_invoice or has_bill):
            continue

        score = 0
        if has_invoice:
            score += 120
        if has_bill:
            score += 70

        if any(tok in n for tok in ["number", "num", "_no", "no_", "code"]):
            score += 90
        if n.endswith("_id") or n == "id" or "_id_" in n:
            score += 50
        if "header" in n:
            score += 20

        if any(tok in n for tok in ["detail", "item", "line", "row"]):
            score -= 140
        if any(tok in n for tok in ["guid", "rid"]):
            score -= 140
        if any(tok in n for tok in ["material", "product", "sku"]):
            score -= 80

        if _is_text_dtype(col_type):
            score += 10
        elif _is_numeric_dtype(col_type):
            score += 5

        if score > best_score:
            best_score = score
            best_col = name

    if best_score < 40:
        return None
    return best_col


def _is_invoice_count_intent(label_text="", sql_text="", prompt_text=""):
    text = f"{label_text} {sql_text} {prompt_text}".lower()
    has_invoice = any(tok in text for tok in ["invoice", "invoices"])
    if not has_invoice:
        return False

    count_like_tokens = [
        "count",
        "distinct",
        "number",
        "how many",
        "no of",
        "no.",
    ]
    return any(tok in text for tok in count_like_tokens)


def _normalize_invoice_count_kpi_plan(kpi_obj, schema_columns, table_name, date_column=None, prompt_text="", logs=None, entity_col=None):
    if not isinstance(kpi_obj, dict):
        return None

    out = dict(kpi_obj)
    label_text = str(out.get("label", "")).strip()
    sql_text = str(out.get("sql") or out.get("value_sql") or "").strip()

    if not _is_invoice_count_intent(label_text=label_text, sql_text=sql_text, prompt_text=prompt_text):
        return out

    chosen_col = entity_col or _select_invoice_entity_key(schema_columns)
    if not chosen_col:
        if logs is not None:
            logs.append(f"[WARN] KPI guard could not identify invoice entity key for '{label_text or 'Invoice Count'}'")
        return out

    chosen_ident = _quote_identifier(chosen_col)
    out["sql"] = f"SELECT COUNT(DISTINCT {chosen_ident}) FROM {table_name}"
    out["value_sql"] = out["sql"]

    if date_column:
        date_ident = _quote_identifier(date_column)
        out["trend_sql"] = (
            f"SELECT CAST({date_ident} AS DATE) AS x, "
            f"COUNT(DISTINCT {chosen_ident}) AS y "
            f"FROM {table_name} GROUP BY 1 ORDER BY 1"
        )
    else:
        out["trend_sql"] = ""

    if logs is not None:
        logs.append(
            f"[GUARD] KPI '{label_text or 'Invoice Count'}' normalized to COUNT(DISTINCT {chosen_col})"
        )
    return out
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
    seen = set()
    for _, row in describe_df.iterrows():
        col_name = str(row[col_name_key]).strip()
        if not col_name or col_name.startswith("#"):
            continue
        key = col_name.lower()
        if key in seen:
            # Databricks DESCRIBE may repeat partition columns; keep first occurrence only.
            continue
        seen.add(key)
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



def _safe_bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
def _databricks_query_row_cap():
    raw = os.getenv("DATABRICKS_QUERY_ROW_CAP", "0")
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def _unquote_databricks_identifier(name):
    return str(name or "").replace("`", "").strip()


def _split_databricks_table_fqn(table_name):
    cleaned = _unquote_databricks_identifier(table_name)
    parts = [p.strip() for p in cleaned.split(".") if p and p.strip()]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    if len(parts) == 1:
        return "", "", parts[0]
    return "", "", ""


def _resolve_related_databricks_table_fqn(base_table, related_table_name):
    raw = str(related_table_name or "").strip()
    if not raw:
        return ""

    if "." in raw:
        parts = [_unquote_databricks_identifier(p) for p in raw.split(".") if str(p).strip()]
        if len(parts) == 3:
            return f"{_quote_identifier(parts[0])}.{_quote_identifier(parts[1])}.{_quote_identifier(parts[2])}"
        if len(parts) == 2:
            return f"{_quote_identifier(parts[0])}.{_quote_identifier(parts[1])}"
        return _quote_identifier(parts[0]) if parts else ""

    catalog = config_databricks_catalog()
    schema = config_databricks_schema()
    if not catalog or not schema:
        parsed_catalog, parsed_schema, _ = _split_databricks_table_fqn(base_table)
        if not catalog:
            catalog = parsed_catalog
        if not schema:
            schema = parsed_schema

    if schema and catalog:
        return f"{_quote_identifier(catalog)}.{_quote_identifier(schema)}.{_quote_identifier(raw)}"
    if schema:
        return f"{_quote_identifier(schema)}.{_quote_identifier(raw)}"
    return _quote_identifier(raw)


def _build_databricks_relationship_virtual_source(connection, base_table, base_columns, include_sample_rows, logs=None):
    relation_specs = [
        {
            "table": "dim_product_master",
            "base_key": "Material_No",
            "dim_key": "ITEM_CBU_CODE",
            "prefix": "product",
        },
        {
            "table": "dim_customer_master",
            "base_key": "AW_CODE",
            "dim_key": "Sold_to_Code",
            "prefix": "customer",
        },
    ]

    base_col_names = [c for c, _ in (base_columns or [])]
    if not base_col_names:
        return None

    select_clauses = []
    schema_columns = []
    used_aliases = set()

    for col_name, col_type in base_columns:
        select_clauses.append(f"f.{_quote_identifier(col_name)} AS {_quote_identifier(col_name)}")
        schema_columns.append((col_name, col_type))
        used_aliases.add(str(col_name).lower())

    join_clauses = []
    joined_tables = []
    alias_idx = 1

    for spec in relation_specs:
        base_key = _find_col_case_insensitive(base_col_names, [spec["base_key"]])
        if not base_key:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: base key {spec['base_key']} not found")
            continue

        dim_fqn = _resolve_related_databricks_table_fqn(base_table, spec["table"])
        if not dim_fqn:
            continue

        try:
            dim_columns = _describe_databricks_table_columns(connection, dim_fqn)
        except Exception as e:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: describe failed ({str(e)})")
            continue

        if not dim_columns:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: no columns returned")
            continue

        dim_col_names = [c for c, _ in dim_columns]
        dim_key = _find_col_case_insensitive(dim_col_names, [spec["dim_key"]])
        if not dim_key:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: dim key {spec['dim_key']} not found")
            continue

        dim_alias = f"d{alias_idx}"
        alias_idx += 1

        join_clauses.append(
            "LEFT JOIN {dim_table} {dim_alias} ON "
            "TRIM(UPPER(CAST(f.{base_key} AS STRING))) = TRIM(UPPER(CAST({dim_alias}.{dim_key} AS STRING)))".format(
                dim_table=dim_fqn,
                dim_alias=dim_alias,
                base_key=_quote_identifier(base_key),
                dim_key=_quote_identifier(dim_key),
            )
        )
        joined_tables.append(dim_fqn)

        for dim_col_name, dim_col_type in dim_columns:
            cleaned = clean_col_name(dim_col_name)
            alias_root = f"{spec['prefix']}_{cleaned}" if cleaned else f"{spec['prefix']}_col"
            alias_name = alias_root
            suffix = 2
            while alias_name.lower() in used_aliases:
                alias_name = f"{alias_root}_{suffix}"
                suffix += 1

            used_aliases.add(alias_name.lower())
            select_clauses.append(
                f"{dim_alias}.{_quote_identifier(dim_col_name)} AS {_quote_identifier(alias_name)}"
            )
            schema_columns.append((alias_name, dim_col_type))

    if not join_clauses:
        return None

    joined_query_source = (
        "(SELECT "
        + ", ".join(select_clauses)
        + f" FROM {base_table} f "
        + " ".join(join_clauses)
        + ") __relation_source"
    )

    schema_context = _load_databricks_schema_context_from_query_source(
        connection,
        joined_query_source,
        include_sample_rows,
        schema_columns,
    )

    if logs is not None:
        logs.append(
            f"[SOURCE] Databricks relationship joins active: base={base_table}, joined={len(joined_tables)}"
        )

    return {
        "base_table": base_table,
        "query_source": joined_query_source,
        "schema_columns": schema_columns,
        "schema_context": schema_context,
        "joined_tables": joined_tables,
    }


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

    enable_relationship_joins = _safe_bool_env("DATABRICKS_ENABLE_RELATION_JOINS", True)
    if enable_relationship_joins:
        relationship_source = _build_databricks_relationship_virtual_source(
            connection,
            base_table,
            base_columns,
            include_sample_rows=include_sample_rows,
            logs=logs,
        )
        if relationship_source:
            return relationship_source

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



def _parse_active_filters_json(active_filters_json):
    if not active_filters_json:
        return {}
    try:
        parsed = json.loads(active_filters_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _resolve_selected_date_column(active_filters_json, date_columns):
    if not date_columns:
        return None

    filters = _parse_active_filters_json(active_filters_json)
    selected_raw = str(filters.get("_date_column") or "").strip()
    if not selected_raw:
        return date_columns[0]

    lookup = {str(c).lower(): c for c in date_columns}
    return lookup.get(selected_raw.lower(), date_columns[0])


def _build_databricks_where_clause(active_filters_json, available_columns, date_column=None, column_type_lookup=None):
    filters = _parse_active_filters_json(active_filters_json)
    if not filters:
        return "", 0

    available = {str(c).lower(): c for c in available_columns}
    col_types = column_type_lookup if isinstance(column_type_lookup, dict) else {}
    where_clauses = []

    start_date = filters.get("_start_date")
    end_date = filters.get("_end_date")
    if date_column:
        date_ident = _quote_identifier(date_column)
        date_dtype = str(col_types.get(str(date_column).lower(), "")).lower()
        date_is_native = ("date" in date_dtype) or ("timestamp" in date_dtype)

        if start_date:
            safe = str(start_date).replace("'", "''")
            if date_is_native:
                where_clauses.append(f"{date_ident} >= DATE '{safe}'")
            else:
                where_clauses.append(f"CAST({date_ident} AS DATE) >= DATE '{safe}'")

        if end_date:
            safe = str(end_date).replace("'", "''")
            if date_is_native:
                # Inclusive end-date while preserving partition pruning opportunities.
                where_clauses.append(f"{date_ident} < DATE_ADD(DATE '{safe}', 1)")
            else:
                where_clauses.append(f"CAST({date_ident} AS DATE) <= DATE '{safe}'")

    for col, val in filters.items():
        if col in {"_start_date", "_end_date", "_date_column"}:
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


def _dedupe_filter_column_candidates(columns):
    out = []
    seen = set()
    for col in (columns or []):
        c = str(col or "").strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _resolve_ranked_filter_columns(raw_candidates, allowed_columns, max_count=6):
    allowed = [str(c).strip() for c in (allowed_columns or []) if str(c).strip()]
    if not allowed:
        return []

    exact_lookup = {c.lower(): c for c in allowed}
    normalized_lookup = {}
    for c in allowed:
        norm = re.sub(r"[^a-z0-9]+", "", c.lower())
        if norm and norm not in normalized_lookup:
            normalized_lookup[norm] = c

    out = []
    seen = set()
    for raw in (raw_candidates or []):
        token = str(raw or "").strip()
        if not token:
            continue

        resolved = exact_lookup.get(token.lower())
        if not resolved:
            norm = re.sub(r"[^a-z0-9]+", "", token.lower())
            resolved = normalized_lookup.get(norm)
        if not resolved:
            continue

        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
        if max_count and len(out) >= max_count:
            break
    return out


def _build_default_filter_defs(filter_columns):
    return [
        {
            "label": col.replace('_', ' ').title(),
            "column": col,
            "values": [],
            "values_loaded": False,
        }
        for col in _dedupe_filter_column_candidates(filter_columns)
    ]


def fetch_filter_values_databricks(column_name, active_filters_json='{}'):
    connection = get_databricks_connection()
    log = []
    try:
        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=False,
            logs=log,
        )
        source_table_query = source_model["query_source"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]

        requested_raw = str(column_name or "").strip()
        if not requested_raw:
            return {"column": "", "values": [], "values_loaded": True, "logs": log, "data_mode": "databricks"}

        lookup = {str(c).lower(): c for c in column_names}
        requested_col = lookup.get(requested_raw.lower())
        if not requested_col:
            return {"column": requested_raw, "values": [], "values_loaded": True, "logs": log, "data_mode": "databricks"}

        parsed_filters = _parse_active_filters_json(active_filters_json)
        if requested_col in parsed_filters:
            parsed_filters.pop(requested_col, None)
        # Also remove case-insensitive matches for safety.
        for k in list(parsed_filters.keys()):
            if str(k).lower() == requested_col.lower():
                parsed_filters.pop(k, None)

        filters_without_current = json.dumps(parsed_filters)
        date_column = _resolve_selected_date_column(filters_without_current, date_cols)
        effective_filters_json, _, _ = _apply_default_date_filters(
            filters_without_current,
            date_column,
            date_range_override=None,
            logs=log,
        )

        where_sql, _ = _build_databricks_where_clause(
            effective_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )

        col_ident = _quote_identifier(requested_col)
        clauses = []
        if where_sql:
            clauses.append(f"({where_sql})")
        clauses.append(f"{col_ident} IS NOT NULL")

        sample_rows = _safe_int_env("DATABRICKS_FILTER_VALUE_SAMPLE_ROWS", 100000)
        value_limit = _safe_int_env("DATABRICKS_FILTER_VALUE_LIMIT", 200)

        base_q = f"SELECT {col_ident} AS raw_v FROM {source_table_query}"
        if clauses:
            base_q += " WHERE " + " AND ".join(clauses)
        if sample_rows > 0:
            base_q += f" LIMIT {sample_rows}"

        sql_text = (
            f"SELECT CAST(raw_v AS STRING) AS v FROM ({base_q}) __fv "
            f"WHERE raw_v IS NOT NULL GROUP BY 1 ORDER BY 1 LIMIT {value_limit}"
        )
        vals_df = fetch_dataframe(connection, sql_text, readonly=True)
        values = [str(v) for v in vals_df["v"].dropna().tolist()] if "v" in vals_df else []

        return {
            "column": requested_col,
            "values": values,
            "values_loaded": True,
            "logs": log,
            "data_mode": "databricks",
        }
    finally:
        connection.close()


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


def _query_uses_relationship_columns(sql_text):
    text = str(sql_text or "").lower()
    return ("product_" in text) or ("customer_" in text)


def _choose_effective_query_source(source_table, query_source, user_sql="", where_sql=""):
    source_base = str(source_table or "").strip()
    source_joined = str(query_source or "").strip()
    if not source_joined or source_joined == source_base:
        return source_base, False

    combined = f"{user_sql or ''}\n{where_sql or ''}"
    needs_join = _query_uses_relationship_columns(combined)
    return (source_joined if needs_join else source_base), needs_join


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

    effective_query_source, using_join_source = _choose_effective_query_source(
        source_table,
        query_source=query_source,
        user_sql=guarded_sql,
        where_sql=where_sql,
    )
    if logs is not None:
        if using_join_source:
            logs.append(f"[PERF] {context}: Using relationship-join source")
        elif query_source and str(query_source).strip() != str(source_table).strip():
            logs.append(f"[PERF] {context}: Using base fact source (join bypass)")

    wrapped_sql = _wrap_sql_with_virtual_views(
        guarded_sql,
        source_table,
        where_sql=where_sql,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
        query_source=effective_query_source,
    )

    df = fetch_dataframe(connection, wrapped_sql, readonly=True)

    if df.empty:
        return df, guarded_sql

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(0)
    return df, guarded_sql



def _execute_databricks_batch_widget_queries(connection, jobs, source_table, where_sql="", query_source=None, logs=None, context="Batch Query"):
    sanitized_jobs = []
    row_cap = _databricks_query_row_cap()

    for job in (jobs or []):
        if not isinstance(job, dict):
            continue
        key = str(job.get("key") or "").strip()
        sql_text = str(job.get("sql") or "").strip()
        ctx = str(job.get("context") or context)
        if not key or not sql_text:
            continue

        normalized_sql = _normalize_databricks_sql_references(
            sql_text,
            source_table,
            view_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        dialect_sql = _normalize_databricks_sql_dialect(normalized_sql)
        guarded_sql, notes = _apply_sql_security_and_cost_guardrails(dialect_sql)

        if logs is not None:
            if dialect_sql != normalized_sql:
                logs.append(f"[SECURITY] {ctx}: Normalized SQL types for Databricks dialect (VARCHAR -> STRING)")
            for note in notes:
                logs.append(f"[SECURITY] {ctx}: {note}")

        sanitized_jobs.append({"key": key, "sql": guarded_sql, "context": ctx})

    if not sanitized_jobs:
        return {}

    any_job_needs_join = _query_uses_relationship_columns(where_sql)
    if not any_job_needs_join:
        any_job_needs_join = any(_query_uses_relationship_columns(job.get("sql", "")) for job in sanitized_jobs)

    base_source, using_join_source = _choose_effective_query_source(
        source_table,
        query_source=query_source,
        user_sql=("\n".join(job.get("sql", "") for job in sanitized_jobs) if any_job_needs_join else ""),
        where_sql=where_sql,
    )
    if logs is not None:
        if using_join_source:
            logs.append(f"[PERF] {context}: Using relationship-join source for batch")
        elif query_source and str(query_source).strip() != str(source_table).strip():
            logs.append(f"[PERF] {context}: Using base fact source for batch (join bypass)")

    base_select = f"SELECT * FROM {base_source}"
    if where_sql:
        base_select += f" WHERE {where_sql}"

    if row_cap > 0:
        base_select = f"SELECT * FROM ({base_select}) __base_capped LIMIT {row_cap}"
        if logs is not None:
            logs.append(f"[PERF] {context}: Applying Databricks row cap DATABRICKS_QUERY_ROW_CAP={row_cap}")

    cte_parts = [f"{DATABRICKS_LOGICAL_VIEW_NAME} AS ({base_select})"]
    union_parts = []
    qid_to_key = {}

    for idx, job in enumerate(sanitized_jobs):
        alias = f"__q{idx}"
        qid = f"q{idx}"
        qid_to_key[qid] = job["key"]
        cte_parts.append(f"{alias} AS ({job['sql']})")
        union_parts.append(f"SELECT '{qid}' AS __qid, to_json(struct(*)) AS __row FROM {alias}")

    batch_sql = "WITH " + ", ".join(cte_parts) + " " + " UNION ALL ".join(union_parts)
    if logs is not None:
        logs.append(f"[PERF] {context}: single-query batch size={len(sanitized_jobs)}")

    raw_df = fetch_dataframe(connection, batch_sql, readonly=True)

    grouped = {job["key"]: [] for job in sanitized_jobs}
    parsed_rows = 0
    matched_rows = 0

    if raw_df is not None and not raw_df.empty:
        raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]
        for _, row in raw_df.iterrows():
            qid = str(row.get("__qid", "")).strip()
            row_blob = row.get("__row")
            if not qid or qid not in qid_to_key:
                continue

            target_key = qid_to_key[qid]
            if row_blob is None or (isinstance(row_blob, float) and pd.isna(row_blob)):
                continue

            if isinstance(row_blob, dict):
                parsed = row_blob
            else:
                try:
                    parsed = json.loads(row_blob if isinstance(row_blob, str) else str(row_blob))
                except Exception:
                    parsed = {}

            if isinstance(parsed, dict):
                grouped[target_key].append(parsed)
                parsed_rows += 1
                matched_rows += 1

        if len(raw_df) > 0 and matched_rows == 0:
            raise ValueError("Batch query returned rows but no widget ids matched; falling back")
        if len(raw_df) > 0 and parsed_rows == 0:
            raise ValueError("Batch query returned rows but parser could not decode any payload rows")

    results = {}
    for job in sanitized_jobs:
        records = grouped.get(job["key"], [])
        df = pd.DataFrame(records) if records else pd.DataFrame()
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(0)
        results[job["key"]] = {
            "ok": True,
            "df": df,
            "executed_sql": job["sql"],
            "logs": [],
            "error": "",
        }

    return results
def _default_custom_chart_plan_from_columns(schema_columns, user_prompt, table_name="final_view"):
    text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
    num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
    date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]

    wants_line = any(token in user_prompt.lower() for token in ["trend", "line", "over time", "monthly", "daily", "weekly", "week", "weekwise"])

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
def generate_viz_config(master_schema_context, debug_logs=None, logical_table_name="master_view"):
    prompt = f"""
    You are a BI Expert. Analyze '{logical_table_name}':
    {master_schema_context}

    STEP 1: FILTERS -> EXACTLY 6 filter columns ranked by usefulness (best first).
    - First 3 are used as default visible filters.
    - First 6 are used as recommended/preloaded filters.
    - Prefer business dimensions suitable for interactive filtering (region, customer, product, channel, status, etc.).
    - Use exact schema column names only.
    STEP 2: KPIS -> EXACTLY 4 KPI SQL queries (each must have label, sql, trend_sql).
    STEP 3: CHARTS -> 6 SQL queries (including heatmap if applicable).

    CRITICAL DATA GRAIN RULE:
    - {logical_table_name} is transaction-level (one row per transaction/event).
    - Master entities (supplier/customer/product/employee etc.) can repeat across many rows.
    - Transaction metrics (revenue, quantity, totals, trends) can aggregate directly on {logical_table_name}.
    - Master attributes (rating, age, salary, static price, static score, etc.) MUST be deduplicated by entity key before AVG or similar stats.
    - For unique entity counts, use COUNT(DISTINCT <detected_entity_key>), not COUNT(*).
    - Dedup pattern example:
      SELECT region, AVG(performance_rating)
      FROM (
          SELECT DISTINCT supplier_id, region, performance_rating
          FROM {logical_table_name}
      ) d
      GROUP BY region

    MANDATORY CHART RULES:
    - Chart 0 (Line/Trend): MUST be a trend over time.
    - Chart 1 (Heatmap): If you have TWO categorical dimensions and a numeric value, create a heatmap. Otherwise, make it a bar chart.
    - Chart 2: No fixed type; choose the best visualization based on the data and question.
    - Charts 3-5: Choose the best visualization for the data.

    TITLE REQUIREMENTS:
    - Titles must clearly mention the metric and dimension.
    KPI REQUIREMENTS:
    - Return EXACTLY 4 KPIs in the kpis array.

    RETURN JSON:
    {{
        "filters": ["Region", "Status", "Category", "Customer_Name", "Product", "BillStatus"],
        "kpis": [
            {{ "label": "Total Spend", "sql": "SELECT SUM(amount) FROM {logical_table_name}", "trend_sql": "SELECT CAST(date_col AS DATE) as x, SUM(amount) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1 LIMIT 7" }},
            {{ "label": "Record Count", "sql": "SELECT COUNT(*) FROM {logical_table_name}", "trend_sql": "SELECT CAST(date_col AS DATE) as x, COUNT(*) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1 LIMIT 7" }}
        ],
        "charts": [
            {{ "title": "Monthly Trend", "type": "line", "sql": "SELECT date_col as x, SUM(val) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1", "xlabel": "Date", "ylabel": "Amount" }},
            {{ "title": "Heatmap", "type": "heatmap", "sql": "SELECT cat1 as x, cat2 as y, SUM(val) as z FROM {logical_table_name} GROUP BY 1,2", "xlabel": "Category 1", "ylabel": "Category 2" }},
            {{ "title": "Top Categories by Value", "type": "bar", "sql": "SELECT CAST(cat1 AS VARCHAR) AS x, SUM(val) AS y FROM {logical_table_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 12", "xlabel": "Category", "ylabel": "Value" }}
        ]
    }}
    """

    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=True, debug_logs=debug_logs, context="Generate Viz Config")
    return (json.loads(res) if res else None), tokens


def generate_additional_kpis(master_schema_context, existing_kpis, needed_count, debug_logs=None, logical_table_name="master_view"):
    try:
        needed = int(needed_count or 0)
    except Exception:
        needed = 0
    if needed <= 0:
        return [], 0

    existing = existing_kpis if isinstance(existing_kpis, list) else []
    existing_json = json.dumps(existing, ensure_ascii=False)

    prompt = f"""
    You are a BI Expert. We already have KPI specs for '{logical_table_name}':
    {existing_json}

    DATA CONTEXT:
    {master_schema_context}

    TASK:
    - Generate EXACTLY {needed} ADDITIONAL KPI specs.
    - These must be different from existing KPIs in metric meaning (not just renamed duplicates).

    RULES:
    1. Return ONLY JSON.
    2. SQL must be SELECT/WITH only and use only {logical_table_name}.
    3. Each KPI must include: label, sql, trend_sql.
    4. trend_sql must return aliases exactly x, y.
    5. For unique entity counts, use COUNT(DISTINCT entity_key), not COUNT(*).
    6. Avoid duplicate labels and duplicate SQL semantics.

    RETURN JSON:
    {{
      "kpis": [
        {{ "label": "KPI 1", "sql": "SELECT ... FROM {logical_table_name}", "trend_sql": "SELECT ... AS x, ... AS y FROM {logical_table_name} ..." }}
      ]
    }}
    """

    res, tokens = call_ai_with_retry(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        debug_logs=debug_logs,
        context="Generate Additional KPIs",
    )

    if not res:
        return [], tokens

    try:
        parsed = json.loads(res)
    except Exception:
        return [], tokens

    raw_kpis = parsed.get("kpis") if isinstance(parsed, dict) else None
    if not isinstance(raw_kpis, list):
        return [], tokens

    cleaned = []
    seen_labels = set(str(k.get("label", "")).strip().lower() for k in existing if isinstance(k, dict))
    for k in raw_kpis:
        if not isinstance(k, dict):
            continue
        label = str(k.get("label", "")).strip()
        sql = str(k.get("sql", "")).strip()
        trend_sql = str(k.get("trend_sql", "")).strip()
        if not label or not sql:
            continue
        key = label.lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        cleaned.append({"label": label, "sql": sql, "trend_sql": trend_sql})
        if len(cleaned) >= needed:
            break

    return cleaned, tokens
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

    wants_line = any(token in user_prompt.lower() for token in ["trend", "line", "over time", "monthly", "daily", "weekly", "week", "weekwise"])

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
    7. If the user explicitly asks for weeks/weekly on x-axis, use weekly grain (for example DATE_TRUNC('week', date_col)) and set xlabel to Week.
    8. final_view is transaction-level. For transactional metrics use final_view directly.
    9. For master/entity attributes (rating, age, salary, static price, scores, etc.), deduplicate by entity key before AVG-style aggregations.
    10. For unique entity questions, use COUNT(DISTINCT <detected_entity_key>) instead of COUNT(*).

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


def _detect_requested_time_grain(user_prompt):
    text = str(user_prompt or "").lower()
    if any(tok in text for tok in ["weekwise", "weekly", " per week", " by week", " week ", "week-on-week", "wow", "w/w"]):
        return "week"
    if any(tok in text for tok in ["monthwise", "monthly", " per month", " by month", " month ", "mom", "m/m"]):
        return "month"
    if any(tok in text for tok in ["daily", " per day", " by day", " day "]):
        return "day"
    return ""


def _enforce_requested_time_grain_on_chart_plan(ai_plan, user_prompt, preferred_date_col=None):
    if not isinstance(ai_plan, dict):
        return ai_plan

    grain = _detect_requested_time_grain(user_prompt)
    if grain != "week":
        return ai_plan

    sql = str(ai_plan.get("sql") or "").strip()
    if not sql:
        return ai_plan

    updated_sql = re.sub(r"(?i)date_trunc\s*\(\s*'month'\s*,", "DATE_TRUNC('week',", sql)
    updated_sql = re.sub(r'(?i)date_trunc\s*\(\s*"month"\s*,', "DATE_TRUNC('week',", updated_sql)

    has_week_bucket = re.search(r"(?i)date_trunc\s*\(\s*'week'\s*,", updated_sql) is not None
    if (not has_week_bucket) and preferred_date_col:
        week_x_expr = f"DATE_TRUNC('week', CAST({_quote_identifier(preferred_date_col)} AS DATE)) AS x"
        updated_sql = re.sub(
            r"(?is)\bselect\b\s*.+?\bas\s+x\s*,\s*(.+?)\bas\s+y\s+\bfrom\b",
            lambda m: f"SELECT {week_x_expr}, {m.group(1).strip()} AS y FROM",
            updated_sql,
            count=1,
        )

    ai_plan["sql"] = updated_sql
    xlabel = str(ai_plan.get("xlabel") or "")
    if not xlabel or "month" in xlabel.lower():
        ai_plan["xlabel"] = "Week"
    title = str(ai_plan.get("title") or "")
    if "month" in title.lower():
        ai_plan["title"] = re.sub(r"(?i)month(ly)?", "Weekly", title)
    return ai_plan



def _safe_date_span_days(start_date, end_date):
    try:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        end_ts = pd.to_datetime(end_date, errors="coerce")
    except Exception:
        return None
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    try:
        days = int((end_ts - start_ts).days) + 1
    except Exception:
        return None
    return max(1, days)


def _maybe_refine_short_window_line_to_weekly(
    connection,
    chart_sql,
    c_data,
    source_table_base,
    source_table_query,
    where_sql,
    date_column,
    applied_start_date,
    applied_end_date,
    logs=None,
    context="Chart",
):
    if not isinstance(c_data, dict):
        return False
    if str(c_data.get("type", "")).lower() != "line":
        return False
    if not date_column:
        return False

    span_days = _safe_date_span_days(applied_start_date, applied_end_date)
    if span_days is None or span_days > 95:
        return False

    old_x = list(c_data.get("x") or [])
    old_y = list(c_data.get("y") or [])
    if len(old_x) >= 4:
        return False

    base_sql = str(chart_sql or "").strip()
    if not base_sql:
        return False

    weekly_plan = _enforce_requested_time_grain_on_chart_plan(
        {
            "sql": base_sql,
            "xlabel": str(c_data.get("xlabel") or ""),
            "title": str(c_data.get("title") or ""),
        },
        user_prompt="weekly",
        preferred_date_col=date_column,
    )
    weekly_sql = str(weekly_plan.get("sql") or "").strip()
    if not weekly_sql or weekly_sql == base_sql:
        return False

    try:
        weekly_df, executed_weekly_sql = _execute_databricks_user_sql(
            connection,
            weekly_sql,
            source_table_base,
            query_source=source_table_query,
            where_sql=where_sql,
            logs=logs,
            context=f"{context} Weekly Refinement",
        )
        if weekly_df.empty or weekly_df.shape[1] < 2:
            return False

        weekly_df.columns = [str(c).lower() for c in weekly_df.columns]
        next_x = weekly_df["x"].tolist() if "x" in weekly_df else weekly_df.iloc[:, 0].tolist()
        next_y_series = weekly_df["y"] if "y" in weekly_df else weekly_df.iloc[:, 1]
        next_y = pd.to_numeric(next_y_series, errors="coerce").fillna(0).astype(float).tolist()

        next_x, next_y = _normalize_and_aggregate_line_series(
            next_x,
            next_y,
            str(c_data.get("title") or ""),
            str(weekly_plan.get("xlabel") or c_data.get("xlabel") or "Week"),
        )
        next_x = [str(v) for v in next_x]

        old_points = min(len(old_x), len(old_y))
        new_points = min(len(next_x), len(next_y))
        if new_points <= old_points or new_points < 4:
            return False

        c_data["x"] = next_x
        c_data["y"] = next_y
        c_data["xlabel"] = str(weekly_plan.get("xlabel") or "Week")
        c_data["sql"] = executed_weekly_sql
        if logs is not None:
            logs.append(f"[INFO] {context}: Applied weekly refinement for short date window ({span_days} days)")
        return True
    except Exception as e:
        if logs is not None:
            logs.append(f"[WARN] {context}: Weekly refinement failed: {str(e)}")
        return False


CUSTOM_CHART_DISAMBIGUATION_STOPWORDS = {
    "show", "plot", "chart", "graph", "visual", "visualize", "give", "create", "generate",
    "for", "from", "with", "using", "into", "onto", "by", "of", "to", "in", "on", "at",
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "this", "that",
    "month", "months", "weekly", "daily", "year", "years", "trend", "total", "count",
    "sum", "average", "avg", "min", "max", "top", "bottom", "highest", "lowest",
    "sales", "revenue", "invoice", "invoices", "amount", "quantity", "value", "values",
    "net", "gross", "bill", "billing", "monthwise", "mom", "number", "numbers", "distinct", "unique","how", "many",
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
        # Ignore generic metric words; only clarify meaningful business terms.
        if term_l in CUSTOM_CHART_DISAMBIGUATION_STOPWORDS:
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
        if term_l in CUSTOM_CHART_DISAMBIGUATION_STOPWORDS:
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


def _is_compact_month_code(value):
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if '-' in text or '/' in text or ':' in text or 'T' in text:
        return False
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split('.', 1)[0]
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) != 6:
        return False
    year = int(digits[:4])
    month = int(digits[4:6])
    return 1900 <= year <= 2100 and 1 <= month <= 12


def _to_week_label_if_possible(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    ts = None
    try:
        ts_candidate = pd.Timestamp(value)
        if not pd.isna(ts_candidate):
            raw = str(value).strip()
            if ('-' in raw) or ('/' in raw) or (':' in raw) or hasattr(value, 'year') or (len(re.sub(r"[^0-9]", "", raw)) >= 8):
                ts = ts_candidate
    except Exception:
        ts = None

    if ts is None:
        return None

    # For week-start timestamps (e.g., Monday), anchor label to mid-week so
    # cross-month weeks are shown under the month users expect in filtered views.
    anchor = ts + pd.Timedelta(days=3)
    week_in_month = ((int(anchor.day) - 1) // 7) + 1
    return f"Week {week_in_month} {anchor.strftime('%b %Y')}"


def _looks_week_bucketed_dates(values):
    if not values:
        return False
    try:
        parsed = pd.to_datetime(pd.Series(list(values)[:260]), errors="coerce", utc=True, format="mixed")
    except TypeError:
        # Older pandas versions do not support format="mixed".
        parsed = pd.Series([pd.to_datetime(v, errors="coerce", utc=True) for v in list(values)[:260]])
    except Exception:
        return False
    parsed = parsed.dropna()
    if parsed.empty or len(parsed) < 3:
        return False
    try:
        parsed = parsed.dt.tz_convert(None)
    except Exception:
        pass
    uniq = sorted(set(pd.Timestamp(v).normalize() for v in parsed.tolist()))
    if len(uniq) < 3:
        return False
    deltas = [(uniq[i + 1] - uniq[i]).days for i in range(len(uniq) - 1)]
    if not deltas:
        return False
    weekly_like = sum(1 for d in deltas if 6 <= int(d) <= 8)
    return weekly_like >= max(2, int(len(deltas) * 0.6))


def _normalize_month_axis_labels(values, title='' , xlabel=''):
    if not values:
        return values

    hint = f"{title} {xlabel}".lower()
    if _looks_week_bucketed_dates(values):
        week_converted = []
        week_count = 0
        for v in values:
            week_label = _to_week_label_if_possible(v)
            if week_label is not None:
                week_converted.append(week_label)
                week_count += 1
            else:
                week_converted.append(v)
        if week_count >= max(2, len(values) // 2):
            return [str(v) for v in week_converted]

    week_hint = ('week' in hint) or ('weekly' in hint) or ('weekwise' in hint)
    if week_hint:
        week_converted = []
        week_count = 0
        for v in values:
            week_label = _to_week_label_if_possible(v)
            if week_label is not None:
                week_converted.append(week_label)
                week_count += 1
            else:
                week_converted.append(v)
        if week_count >= max(2, len(values) // 2):
            return [str(v) for v in week_converted]

    force_month = ('month' in hint) or ('monthly' in hint) or ('yyyy_mm' in hint)
    avoid_month = any(tok in hint for tok in ['week', 'weekly', 'day', 'daily', 'quarter', 'qtr', 'year', 'yearly'])
    if avoid_month and not force_month:
        return values

    converted = []
    converted_count = 0
    for v in values:
        month_label = _to_month_label_if_possible(v)
        if month_label is not None:
            converted.append(month_label)
            converted_count += 1
        else:
            converted.append(v)

    compact_month_count = sum(1 for v in values if _is_compact_month_code(v))
    if force_month or (
        compact_month_count >= max(2, len(values) // 2)
        and converted_count >= max(2, len(values) // 2)
    ):
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


def _value_has_datetime_hint(value):
    s = str(value or "").strip()
    if not s:
        return False

    sl = s.lower()
    if any(ch in s for ch in ['-', '/', ':', 'T']):
        return True
    if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", sl):
        return True

    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 6:
        try:
            y = int(digits[:4])
            m = int(digits[4:6])
            return 1900 <= y <= 2100 and 1 <= m <= 12
        except Exception:
            return False
    if len(digits) == 8:
        try:
            y = int(digits[:4])
            m = int(digits[4:6])
            d = int(digits[6:8])
            return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31
        except Exception:
            return False

    return False


def _series_looks_datetime(values):
    if not values:
        return False

    sample = [v for v in list(values)[:300] if str(v).strip()]
    if not sample:
        return False

    hint_count = sum(1 for v in sample if _value_has_datetime_hint(v))
    if hint_count < max(2, int(len(sample) * 0.3)):
        return False

    try:
        parsed = pd.to_datetime(pd.Series(sample), errors="coerce", utc=True, format="mixed")
    except TypeError:
        # Older pandas versions do not support format="mixed".
        parsed = pd.Series([pd.to_datetime(v, errors="coerce", utc=True) for v in sample])
    valid_count = int(parsed.notna().sum())
    return valid_count >= max(2, int(len(sample) * 0.6))


def _coerce_categorical_line_to_bar(c_data, chart_type, max_points=24):
    if str(chart_type).lower() != "line":
        return chart_type

    x_vals = list(c_data.get("x") or [])
    y_vals = list(c_data.get("y") or [])
    if not x_vals or not y_vals:
        return chart_type

    if _series_looks_datetime(x_vals):
        return chart_type

    if len(x_vals) <= max_points:
        return chart_type

    aggregated = {}
    order = []
    for x_val, y_val in zip(x_vals, y_vals):
        key = str(x_val)
        if key not in aggregated:
            aggregated[key] = 0.0
            order.append(key)
        try:
            aggregated[key] += float(y_val)
        except Exception:
            pass

    pairs = [(k, aggregated[k]) for k in order]

    hint = f"{c_data.get('title', '')} {c_data.get('xlabel', '')}".lower()
    if "distribution" in hint or "top" in hint or "by" in hint:
        pairs.sort(key=lambda row: row[1], reverse=True)

    if len(pairs) > max_points:
        kept = pairs[: max_points - 1]
        others_sum = float(sum(v for _, v in pairs[max_points - 1 :]))
        if abs(others_sum) > 0:
            kept.append(("Others", others_sum))
        pairs = kept

    c_data["x"] = [x for x, _ in pairs]
    c_data["y"] = [y for _, y in pairs]
    c_data["type"] = "bar"
    return "bar"
def _rebucket_numeric_distribution(x_values, y_values, title="", xlabel="", ylabel="", max_points=60, max_bins=20):
    try:
        x_vals = list(x_values or [])
        y_vals = list(y_values or [])
    except Exception:
        return x_values, y_values, False

    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        return x_values, y_values, False

    if _series_looks_datetime(x_vals):
        return x_values, y_values, False

    x_num = pd.to_numeric(pd.Series(x_vals), errors="coerce")
    valid_ratio = float(x_num.notna().mean()) if len(x_num) else 0.0
    if valid_ratio < 0.9:
        return x_values, y_values, False

    unique_count = int(x_num.nunique(dropna=True))
    hint = f"{title} {xlabel} {ylabel}".lower()
    distribution_hint = any(tok in hint for tok in ["distribution", "hist", "frequency"])

    if unique_count <= max_points and not distribution_hint:
        return x_values, y_values, False

    y_num = pd.to_numeric(pd.Series(y_vals), errors="coerce").fillna(0.0)
    valid_mask = x_num.notna() & y_num.notna()
    if int(valid_mask.sum()) < 3:
        return x_values, y_values, False

    xv = x_num[valid_mask].astype(float).to_numpy()
    yv = y_num[valid_mask].astype(float).to_numpy()

    x_min = float(np.nanmin(xv))
    x_max = float(np.nanmax(xv))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return x_values, y_values, False

    bin_count = min(int(max_bins), max(8, int(np.sqrt(max(unique_count, 1)))))
    bin_count = max(2, bin_count)

    edges = np.linspace(x_min, x_max, bin_count + 1)
    hist, edges = np.histogram(xv, bins=edges, weights=yv)

    labels = []
    values = []
    for i in range(len(hist)):
        v = float(hist[i])
        if abs(v) < 1e-12:
            continue
        lo = float(edges[i])
        hi = float(edges[i + 1])
        labels.append(f"{lo:,.2f} to {hi:,.2f}")
        values.append(v)

    if not labels:
        return x_values, y_values, False

    return labels, values, True
def _extract_kpi_sparkline_from_df(df, target_points=None):
    if df is None or df.empty or df.shape[1] < 2:
        return []

    work = pd.DataFrame({
        "x": df.iloc[:, 0],
        "y": pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
    })
    ts_work = None

    # Prefer chronological sorting when x looks like date/timestamp.
    parsed_ts = pd.to_datetime(work["x"], errors="coerce", utc=True)
    try:
        parsed_ts = parsed_ts.dt.tz_convert(None)
    except Exception:
        pass
    if parsed_ts.notna().sum() >= max(2, len(work) // 2):
        ts_work = (
            work.assign(_ts=parsed_ts)
            .dropna(subset=["_ts"])
            .sort_values("_ts", kind="mergesort")
        )
        if ts_work.empty:
            return []

        min_ts = ts_work["_ts"].min()
        max_ts = ts_work["_ts"].max()
        span_days = max(1, int((max_ts - min_ts).days) + 1) if pd.notna(min_ts) and pd.notna(max_ts) else 1

        bucket_kind = "month"
        if span_days <= 45:
            bucket_kind = "day"
            period_series = ts_work["_ts"].dt.floor("D")
        elif span_days <= 180:
            bucket_kind = "week"
            period_series = ts_work["_ts"].dt.to_period("W").dt.start_time
        else:
            period_series = ts_work["_ts"].dt.to_period("M").dt.to_timestamp()

        work = (
            ts_work.assign(_period=period_series)
            .groupby("_period", as_index=False)["y"]
            .sum()
            .sort_values("_period", kind="mergesort")
        )

        # Drop only an incomplete current period when doing so still keeps at least 2 points.
        if len(work) >= 3:
            _now_utc = pd.Timestamp.now(tz="UTC").tz_convert(None)
            if bucket_kind == "day":
                current_period_start = _now_utc.floor("D")
            elif bucket_kind == "week":
                current_period_start = _now_utc.to_period("W").start_time
            else:
                current_period_start = _now_utc.normalize().replace(day=1)
            if work.iloc[-1]["_period"] == current_period_start and (len(work) - 1) >= 2:
                work = work.iloc[:-1]
    else:
        # Fallback to numeric sorting when possible.
        parsed_num = pd.to_numeric(work["x"], errors="coerce")
        if parsed_num.notna().sum() >= max(2, len(work) // 2):
            work = work.assign(_k=parsed_num).sort_values("_k", kind="mergesort")

    values = [float(v) for v in work["y"].tolist()]
    if not values:
        return []
    if target_points is not None:
        try:
            tp = int(target_points)
        except Exception:
            tp = 0
        if tp > 0 and len(values) > tp:
            values = values[-tp:]

    # If coarse bucketing collapsed to a single point, retry with day buckets.
    if len(values) < 2 and ts_work is not None and not ts_work.empty:
        day_work = (
            ts_work.assign(_day=ts_work["_ts"].dt.floor("D"))
            .groupby("_day", as_index=False)["y"]
            .sum()
            .sort_values("_day", kind="mergesort")
        )
        alt_values = [float(v) for v in day_work["y"].tolist()]
        if len(alt_values) >= 2:
            values = alt_values
            if target_points is not None:
                try:
                    tp = int(target_points)
                except Exception:
                    tp = 0
                if tp > 0 and len(values) > tp:
                    values = values[-tp:]
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
            y_series = df["y"] if "y" in df else df.iloc[:, 1]
            chart_data["y"] = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(float).tolist()
            if chart_type == "line":
                chart_data["x"], chart_data["y"] = _normalize_and_aggregate_line_series(
                    chart_data["x"], chart_data["y"], chart_data.get("title", ""), chart_data.get("xlabel", "")
                )
            else:
                chart_data["x"] = _normalize_month_axis_labels(chart_data["x"], chart_data.get("title", ""), chart_data.get("xlabel", ""))

            chart_data["x"], chart_data["y"], _ = _rebucket_numeric_distribution(
                chart_data.get("x", []),
                chart_data.get("y", []),
                title=chart_data.get("title", ""),
                xlabel=chart_data.get("xlabel", ""),
                ylabel=chart_data.get("ylabel", ""),
            )

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
        sparkline_data = [base] * KPI_TREND_POINTS

    return {
        "label": label,
        "value": _format_kpi_display_value(value_raw),
        "sparkline": [float(v) for v in sparkline_data],
    }


def generate_custom_kpi_from_prompt_databricks(user_prompt, active_filters_json='{}', clarification_choice=None, date_range_override=None, allow_ambiguity_fallback=False):
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
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        effective_filters_json, applied_start_date, applied_end_date = _apply_default_date_filters(
            active_filters_json,
            date_column,
            date_range_override=date_range_override,
            logs=llm_logs,
        )

        where_sql, filter_count = _build_databricks_where_clause(
            effective_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )
        if filter_count > 0:
            llm_logs.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        normalized_choice = _normalize_custom_chart_clarification_choice(clarification_choice, schema_columns)
        ambiguity = None
        if not allow_ambiguity_fallback:
            ambiguity = _detect_custom_chart_ambiguity(
                connection,
                user_prompt,
                schema_columns,
                source_table_query,
                where_sql=where_sql,
                clarification_choice=normalized_choice,
                logs=llm_logs,
            )
        else:
            llm_logs.append("[FALLBACK] Ambiguity fallback enabled for custom KPI: letting LLM choose best match")

        if ambiguity:
            return {
                "needs_clarification": True,
                "clarification": ambiguity,
                "tokens_used": 0,
                "logs": llm_logs,
                "data_mode": "databricks",
            }

        prompt_for_kpi = _apply_custom_chart_clarification_to_prompt(user_prompt, normalized_choice, logs=llm_logs)

        ai_plan, tokens_used = generate_custom_kpi_plan(
            schema_context,
            prompt_for_kpi,
            debug_logs=llm_logs,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        if not ai_plan:
            ai_plan = _default_custom_kpi_plan_from_columns(
                schema_columns,
                user_prompt,
                table_name=DATABRICKS_LOGICAL_VIEW_NAME,
            )

        ai_plan = _normalize_invoice_count_kpi_plan(
            ai_plan,
            schema_columns,
            DATABRICKS_LOGICAL_VIEW_NAME,
            date_column=date_column,
            prompt_text=prompt_for_kpi,
            logs=llm_logs,
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
            ai_plan = _normalize_invoice_count_kpi_plan(
                ai_plan,
                schema_columns,
                DATABRICKS_LOGICAL_VIEW_NAME,
                date_column=date_column,
                prompt_text=prompt_for_kpi,
                logs=llm_logs,
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
                        sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=KPI_TREND_POINTS)
                except Exception as e:
                    llm_logs.append(f"[WARN] KPI trend query failed: {str(e)}")

        if not sparkline_data:
            try:
                base = float(value_raw)
            except Exception:
                base = 0.0
            sparkline_data = [base] * KPI_TREND_POINTS

        kpi_payload = _build_custom_kpi_payload(ai_plan, value_raw, sparkline_data)
        kpi_payload["sql"] = executed_value_sql
        kpi_payload["trend_sql"] = executed_trend_sql or trend_sql
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


def execute_dashboard_logic_databricks(
    active_filters_json=None,
    session_id=None,
    filters_override=None,
    date_range_override=None,
):
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
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}

        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
        num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        effective_filters_json, applied_start_date, applied_end_date = _apply_default_date_filters(
            active_filters_json,
            date_column,
            date_range_override=date_range_override,
            logs=log,
        )

        where_sql, filter_count = _build_databricks_where_clause(
            effective_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )
        if filter_count > 0:
            log.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        date_range = {"min": None, "max": None}
        cached_date_range_used = False
        if isinstance(date_range_override, dict):
            min_override = date_range_override.get("min")
            max_override = date_range_override.get("max")
            if min_override not in (None, "") or max_override not in (None, ""):
                date_range = {
                    "min": str(min_override) if min_override not in (None, "") else None,
                    "max": str(max_override) if max_override not in (None, "") else None,
                }
                cached_date_range_used = True
                log.append("[CACHE] Reusing cached date range")
        if date_column and not cached_date_range_used:
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

        if _is_databricks_mode_active() and STRICT_SQL_GUARDRAILS:
            log.append(
                f"[SECURITY] Databricks SQL guardrails active (SELECT/WITH only, forbidden keywords blocked, max LIMIT={AI_SQL_MAX_LIMIT})"
            )

        plan, tokens = generate_viz_config(
            schema_context,
            debug_logs=log,
            logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        total_tokens += tokens

        if isinstance(plan, dict):
            llm_ranked_filters = _resolve_ranked_filter_columns(
                plan.get("filters", []),
                text_cols,
                max_count=6,
            )
            if llm_ranked_filters:
                plan["filters"] = llm_ranked_filters
                log.append(f"[FILTER] LLM-ranked filters: {', '.join(llm_ranked_filters)}")
            else:
                plan["filters"] = []
                log.append("[FILTER] LLM-ranked filters unavailable; falling back to metadata columns")

        if isinstance(plan, dict):
            plan_kpis = plan.get("kpis", [])
            if not isinstance(plan_kpis, list):
                plan_kpis = []
            if len(plan_kpis) < 4:
                needed_kpis = 4 - len(plan_kpis)
                extra_kpis, extra_tokens = generate_additional_kpis(
                    schema_context,
                    plan_kpis,
                    needed_kpis,
                    debug_logs=log,
                    logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                )
                total_tokens += extra_tokens
                if extra_kpis:
                    plan["kpis"] = plan_kpis + extra_kpis

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
                "domain": "General",
                "filters": text_cols[:6],
                "kpis": [
                    {"label": "Record Count", "sql": f"SELECT COUNT(*) FROM {DATABRICKS_LOGICAL_VIEW_NAME}"},
                ],
                "charts": [
                    {"title": "Overview", "type": "line" if time_col else "bar", "sql": fallback_sql, "xlabel": "", "ylabel": ""}
                ],
            }
        theme = FIXED_DASHBOARD_THEME

        output = {
            "domain": "General",
            "theme": theme,
            "filters": [],
            "kpis": [],
            "charts": [],
            "logs": log,
            "date_range": date_range,
            "date_columns": date_cols,
            "selected_date_column": date_column,
            "applied_start_date": applied_start_date,
            "applied_end_date": applied_end_date,
            "has_date_column": bool(date_cols),
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

        lazy_filter_values = _safe_bool_env("DATABRICKS_FILTER_LAZY_VALUES", True)

        cached_filters = []
        if isinstance(filters_override, list):
            for f in filters_override:
                if not isinstance(f, dict):
                    continue
                col = str(f.get("column", "")).strip()
                if not col:
                    continue
                vals = f.get("values", [])
                if not isinstance(vals, list):
                    vals = []
                cleaned_vals = [str(v) for v in vals if str(v).strip()]
                cached_filters.append({
                    "label": str(f.get("label") or col.replace('_', ' ').title()),
                    "column": col,
                    "values": cleaned_vals,
                    "values_loaded": bool(f.get("values_loaded")) or bool(cleaned_vals),
                })

        if lazy_filter_values:
            if cached_filters:
                output["filters"] = cached_filters
                log.append(f"[CACHE] Reusing cached filter definitions ({len(cached_filters)})")
            else:
                filter_columns = _dedupe_filter_column_candidates((plan.get("filters", []) + text_cols))
                output["filters"] = _build_default_filter_defs(filter_columns)
                log.append(f"[PERF] Lazy filter values enabled ({len(output['filters'])} filter columns)")
        elif cached_filters:
            output["filters"] = [f for f in cached_filters if f.get("values")]
            log.append(f"[CACHE] Reusing cached filter definitions ({len(output['filters'])})")
        else:
            filter_candidates = []
            seen = set()
            max_filter_candidates = _safe_int_env("DATABRICKS_FILTER_CANDIDATE_LIMIT", 12)
            for col in (plan.get("filters", []) + text_cols):
                c = str(col)
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                filter_candidates.append(c)
                if len(filter_candidates) >= max_filter_candidates:
                    break

            if len(plan.get("filters", []) + text_cols) > len(filter_candidates):
                log.append(f"[PERF] Filter candidates capped to {len(filter_candidates)} columns")

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
                    output["filters"].append({
                        "label": col.replace('_', ' ').title(),
                        "column": col,
                        "values": vals,
                        "values_loaded": True,
                    })
                except Exception:
                    continue

        def _run_kpi_spec(label, kpi_sql, trend_sql=None, context_prefix="KPI"):
            label_text = str(label or "Metric").strip() or "Metric"
            kpi_sql = str(kpi_sql or "").replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
            if not kpi_sql:
                raise ValueError("Empty KPI SQL")

            df, executed_value_sql = _execute_databricks_user_sql(
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
            executed_trend_sql = ""
            if trend_sql:
                trend_sql = str(trend_sql).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
                if trend_sql:
                    try:
                        trend_df, executed_trend_sql = _execute_databricks_user_sql(
                            connection,
                            trend_sql,
                            source_table_base,
                            query_source=source_table_query,
                            where_sql=where_sql,
                            logs=log,
                            context=f"{context_prefix} Trend {label_text}",
                        )
                        if not trend_df.empty and trend_df.shape[1] >= 2:
                            sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=KPI_TREND_POINTS)
                    except Exception as e:
                        log.append(f"[WARN] KPI trend query failed for {label_text}: {str(e)}")

            if not sparkline_data:
                try:
                    base_val = float(val)
                except Exception:
                    base_val = 0.0
                sparkline_data = [max(0.0, base_val)] * KPI_TREND_POINTS

            return {
                "label": label_text,
                "value": _format_kpi_display_value(val),
                "sparkline": [float(v) for v in sparkline_data],
                "sql": executed_value_sql,
                "trend_sql": executed_trend_sql or trend_sql or "",
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
        invoice_entity_col = _select_invoice_entity_key(schema_columns)
        if invoice_entity_col:
            log.append(f"[GUARD] Invoice entity key selected: {invoice_entity_col}")

        def _enforce_invoice_count_kpi(kpi_obj):
            return _normalize_invoice_count_kpi_plan(
                kpi_obj,
                schema_columns,
                DATABRICKS_LOGICAL_VIEW_NAME,
                date_column=date_column,
                logs=log,
                entity_col=invoice_entity_col,
            )

        kpis = []
        for _k in (plan.get("kpis", []) or []):
            norm = _enforce_invoice_count_kpi(_k)
            if norm is not None:
                kpis.append(norm)

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

        fallback_kpis = [
            {
                "label": "Record Count",
                "sql": f"SELECT COUNT(*) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, COUNT(*) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
            },
            {
                "label": "Distinct Invoices",
                "sql": f"SELECT COUNT(DISTINCT {_quote_identifier(invoice_entity_col)}) FROM {DATABRICKS_LOGICAL_VIEW_NAME}",
                "trend_sql": f"SELECT CAST({_quote_identifier(date_column)} AS DATE) AS x, COUNT(DISTINCT {_quote_identifier(invoice_entity_col)}) AS y FROM {DATABRICKS_LOGICAL_VIEW_NAME} GROUP BY 1 ORDER BY 1" if date_column else "",
            } if invoice_entity_col else None,
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
                valid_kpis.append({
                    "label": "No Data",
                    "value": "-",
                    "sparkline": [0.0] * KPI_TREND_POINTS,
                    "sql": "",
                    "trend_sql": "",
                })
            else:
                valid_kpis.append({
                    "label": f"Rows (Filtered) {len(valid_kpis)+1}",
                    "value": _format_kpi_display_value(filtered_row_count),
                    "sparkline": [float(filtered_row_count)] * KPI_TREND_POINTS,
                    "sql": "",
                    "trend_sql": "",
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
                    fallback_df, fallback_executed_sql = _execute_databricks_user_sql(
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
                    c_data["sql"] = fallback_executed_sql
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
                "sql": "",
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
                df, executed_chart_sql = _execute_databricks_user_sql(
                    connection,
                    chart_sql,
                    source_table_base,
                    query_source=source_table_query,
                    where_sql=where_sql,
                    logs=log,
                    context=f"Chart {i}",
                )
                c_data["sql"] = executed_chart_sql

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
                                c_data["x"] = [str(v) for v in c_data.get("x", [])]
                                _maybe_refine_short_window_line_to_weekly(
                                    connection=connection,
                                    chart_sql=chart_sql,
                                    c_data=c_data,
                                    source_table_base=source_table_base,
                                    source_table_query=source_table_query,
                                    where_sql=where_sql,
                                    date_column=date_column,
                                    applied_start_date=applied_start_date,
                                    applied_end_date=applied_end_date,
                                    logs=log,
                                    context=f"Chart {i}",
                                )
                            else:
                                c_data["x"] = _normalize_month_axis_labels(c_data["x"], c_data.get("title", ""), c_data.get("xlabel", ""))

                            c_data["x"], c_data["y"], rebucketed = _rebucket_numeric_distribution(
                                c_data.get("x", []),
                                c_data.get("y", []),
                                title=c_data.get("title", ""),
                                xlabel=c_data.get("xlabel", ""),
                                ylabel=c_data.get("ylabel", ""),
                            )
                            if rebucketed:
                                log.append(f"[INFO] Chart {i} rebucketed numeric distribution for readability")

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
                "sql": "",
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

        output["__cache"] = {
            "filters": output.get("filters", []),
            "source_table": source_table_base,
            "date_range": output.get("date_range"),
            "selected_date_column": output.get("selected_date_column"),
        }
        return output
    finally:
        connection.close()





def execute_dashboard_filter_refresh_databricks(
    active_filters_json=None,
    widget_state=None,
    session_id=None,
    filters_override=None,
    date_range_override=None,
):
    log = []
    if not session_id:
        session_id = str(uuid.uuid4())

    widget_state = widget_state if isinstance(widget_state, dict) else {}
    kpi_specs = widget_state.get("kpis", [])
    if not isinstance(kpi_specs, list):
        kpi_specs = []
    chart_specs = widget_state.get("charts", [])
    if not isinstance(chart_specs, list):
        chart_specs = []

    connection = get_databricks_connection()
    try:
        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=False,
            logs=log,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        effective_filters_json, applied_start_date, applied_end_date = _apply_default_date_filters(
            active_filters_json,
            date_column,
            date_range_override=date_range_override,
            logs=log,
        )


        where_sql, filter_count = _build_databricks_where_clause(
            effective_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )
        if filter_count > 0:
            log.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        date_range = {"min": None, "max": None}
        cached_date_range_used = False
        if isinstance(date_range_override, dict):
            min_override = date_range_override.get("min")
            max_override = date_range_override.get("max")
            if min_override not in (None, "") or max_override not in (None, ""):
                date_range = {
                    "min": str(min_override) if min_override not in (None, "") else None,
                    "max": str(max_override) if max_override not in (None, "") else None,
                }
                cached_date_range_used = True
                log.append("[CACHE] Reusing cached date range")

        if date_column and not cached_date_range_used:
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

        output = {
            "domain": "General",
            "theme": FIXED_DASHBOARD_THEME,
            "filters": [],
            "kpis": [],
            "charts": [],
            "logs": log,
            "date_range": date_range,
            "date_columns": date_cols,
            "selected_date_column": date_column,
            "applied_start_date": applied_start_date,
            "applied_end_date": applied_end_date,
            "has_date_column": bool(date_cols),
            "tokens_used": 0,
            "master_preview": None,
            "data_mode": "databricks",
            "source_table": source_table_base,
            "session_id": session_id,
        }

        lazy_filter_values = _safe_bool_env("DATABRICKS_FILTER_LAZY_VALUES", True)
        cached_filters = []
        if isinstance(filters_override, list):
            for f in filters_override:
                if not isinstance(f, dict):
                    continue
                col = str(f.get("column", "")).strip()
                if not col:
                    continue
                vals = f.get("values", [])
                if not isinstance(vals, list):
                    vals = []
                cleaned_vals = [str(v) for v in vals if str(v).strip()]
                cached_filters.append({
                    "label": str(f.get("label") or col.replace('_', ' ').title()),
                    "column": col,
                    "values": cleaned_vals,
                    "values_loaded": bool(f.get("values_loaded")) or bool(cleaned_vals),
                })

        if lazy_filter_values:
            if cached_filters:
                output["filters"] = cached_filters
                log.append(f"[CACHE] Reusing cached filter definitions ({len(cached_filters)})")
            else:
                output["filters"] = _build_default_filter_defs(text_cols)
                log.append(f"[PERF] Lazy filter values enabled ({len(output['filters'])} filter columns)")
        elif cached_filters:
            output["filters"] = [f for f in cached_filters if f.get("values")]
            log.append(f"[CACHE] Reusing cached filter definitions ({len(output['filters'])})")

        # Build a unique query plan first, then execute in parallel.
        unique_queries = {}

        def _register_query(sql_text, context):
            sql_key = str(sql_text or "").strip()
            if not sql_key:
                return ""
            if sql_key not in unique_queries:
                unique_queries[sql_key] = {
                    "sql": sql_key,
                    "context": context,
                }
            return sql_key

        kpi_runtime = []
        for idx, spec in enumerate(kpi_specs):
            if not isinstance(spec, dict):
                continue
            label = str(spec.get("label") or f"KPI {idx+1}").strip() or f"KPI {idx+1}"
            value_sql = str(spec.get("sql") or spec.get("value_sql") or "").strip()
            trend_sql = str(spec.get("trend_sql") or "").strip()
            value_sql = value_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
            trend_sql = trend_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)

            if not value_sql:
                raw_spark = spec.get("sparkline")
                if not isinstance(raw_spark, list) or not raw_spark:
                    raw_spark = [0.0] * KPI_TREND_POINTS
                output["kpis"].append({
                    "label": label,
                    "value": str(spec.get("value") or "-"),
                    "sparkline": [float(v) for v in raw_spark],
                    "sql": "",
                    "trend_sql": trend_sql,
                })
                continue

            value_key = _register_query(value_sql, f"Filter KPI {label}")
            trend_key = _register_query(trend_sql, f"Filter KPI Trend {label}") if trend_sql else ""
            kpi_runtime.append({
                "label": label,
                "value_key": value_key,
                "trend_key": trend_key,
                "trend_sql": trend_sql,
                "source_spec": spec,
            })

        chart_runtime = []
        for idx, spec in enumerate(chart_specs):
            if not isinstance(spec, dict):
                continue
            chart_id = str(spec.get("id") or f"chart_{idx}")
            chart_type = str(spec.get("type") or "bar").lower()
            if chart_type not in ALLOWED_CUSTOM_CHART_TYPES:
                chart_type = "bar"
            chart_sql = str(spec.get("sql") or "").strip()
            chart_sql = chart_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)

            if not chart_sql:
                output["charts"].append({
                    "id": chart_id,
                    "title": str(spec.get("title") or f"Chart {idx+1}"),
                    "type": chart_type,
                    "xlabel": str(spec.get("xlabel") or ""),
                    "ylabel": str(spec.get("ylabel") or ""),
                    "sql": "",
                    "x": list(spec.get("x") or []),
                    "y": list(spec.get("y") or []),
                    "z": list(spec.get("z") or []),
                    "columns": spec.get("columns") if isinstance(spec.get("columns"), list) else [],
                    "rows": spec.get("rows") if isinstance(spec.get("rows"), list) else [],
                    "labels": spec.get("labels") if isinstance(spec.get("labels"), list) else [],
                    "parents": spec.get("parents") if isinstance(spec.get("parents"), list) else [],
                    "values": list(spec.get("values") or []),
                    "measure": spec.get("measure") if isinstance(spec.get("measure"), list) else [],
                    "showDataLabels": bool(spec.get("showDataLabels", False)),
                })
                continue

            query_key = _register_query(chart_sql, f"Filter Chart {chart_id}")
            chart_runtime.append({
                "chart_id": chart_id,
                "title": str(spec.get("title") or f"Chart {idx+1}"),
                "type": chart_type,
                "xlabel": str(spec.get("xlabel") or ""),
                "ylabel": str(spec.get("ylabel") or ""),
                "showDataLabels": bool(spec.get("showDataLabels", False)),
                "query_key": query_key,
            })

        def _execute_single_filter_query(job_sql, job_context):
            local_logs = []
            local_con = get_databricks_connection()
            try:
                df, executed_sql = _execute_databricks_user_sql(
                    local_con,
                    job_sql,
                    source_table_base,
                    query_source=source_table_query,
                    where_sql=where_sql,
                    logs=local_logs,
                    context=job_context,
                )
                return {
                    "ok": True,
                    "df": df,
                    "executed_sql": executed_sql,
                    "logs": local_logs,
                    "error": "",
                }
            except Exception as e:
                return {
                    "ok": False,
                    "df": pd.DataFrame(),
                    "executed_sql": job_sql,
                    "logs": local_logs,
                    "error": str(e),
                }
            finally:
                try:
                    local_con.close()
                except Exception:
                    pass

        query_results = {}
        if unique_queries:
            use_single_query = _safe_bool_env("DATABRICKS_FILTER_SINGLE_QUERY", True)
            if use_single_query and len(unique_queries) > 1:
                batch_jobs = [
                    {"key": key, "sql": meta["sql"], "context": meta["context"]}
                    for key, meta in unique_queries.items()
                ]
                try:
                    query_results = _execute_databricks_batch_widget_queries(
                        connection,
                        batch_jobs,
                        source_table_base,
                        where_sql=where_sql,
                        query_source=source_table_query,
                        logs=log,
                        context="Filter Refresh Batch",
                    )
                except Exception as e:
                    log.append(f"[WARN] Filter single-query execution failed; falling back to fanout: {str(e)}")
                    query_results = {}

            if query_results:
                has_non_empty = any(
                    bool(res.get("ok")) and isinstance(res.get("df"), pd.DataFrame) and (not res.get("df").empty)
                    for res in query_results.values()
                )
                if not has_non_empty:
                    log.append("[WARN] Filter single-query returned no rows for all widgets; falling back to fanout")
                    query_results = {}

            if not query_results:
                max_workers = _safe_int_env("DATABRICKS_FILTER_MAX_WORKERS", 4)
                max_workers = max(1, min(max_workers, len(unique_queries)))
                log.append(f"[PERF] Filter refresh query fanout={len(unique_queries)} workers={max_workers}")

                if max_workers == 1:
                    for key, meta in unique_queries.items():
                        res = _execute_single_filter_query(meta["sql"], meta["context"])
                        query_results[key] = res
                        for line in res.get("logs", []):
                            log.append(line)
                        if not res.get("ok"):
                            log.append("[WARN] Filter query failed ({0}): {1}".format(meta["context"], res.get("error")))
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map = {
                            executor.submit(_execute_single_filter_query, meta["sql"], meta["context"]): (key, meta)
                            for key, meta in unique_queries.items()
                        }
                        for future in as_completed(future_map):
                            key, meta = future_map[future]
                            try:
                                res = future.result()
                            except Exception as e:
                                res = {
                                    "ok": False,
                                    "df": pd.DataFrame(),
                                    "executed_sql": meta["sql"],
                                    "logs": [],
                                    "error": str(e),
                                }
                            query_results[key] = res
                            for line in res.get("logs", []):
                                log.append(line)
                            if not res.get("ok"):
                                log.append("[WARN] Filter query failed ({0}): {1}".format(meta["context"], res.get("error")))
        # Build KPI outputs in the same order as provided widget state.
        for item in kpi_runtime:
            label = item["label"]
            value_res = query_results.get(item["value_key"], None)
            if not value_res or not value_res.get("ok"):
                output["kpis"].append({
                    "label": label,
                    "_refresh_failed": True,
                    "sql": item.get("value_key", ""),
                    "trend_sql": item.get("trend_sql", ""),
                })
                continue

            value_df = value_res.get("df")
            value_raw = _extract_first_scalar(value_df, default=0.0)

            sparkline_data = []
            trend_sql_out = item.get("trend_sql", "")
            if item.get("trend_key"):
                trend_res = query_results.get(item["trend_key"], None)
                if trend_res and trend_res.get("ok"):
                    trend_df = trend_res.get("df")
                    if trend_df is not None and not trend_df.empty:
                        sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=KPI_TREND_POINTS)
                    trend_sql_out = trend_res.get("executed_sql") or trend_sql_out

            if not sparkline_data:
                try:
                    base = float(value_raw)
                except Exception:
                    base = 0.0
                sparkline_data = [base] * KPI_TREND_POINTS

            output["kpis"].append({
                "label": label,
                "value": _format_kpi_display_value(value_raw),
                "sparkline": [float(v) for v in sparkline_data],
                "sql": value_res.get("executed_sql") or "",
                "trend_sql": trend_sql_out,
            })

        # Build chart outputs in order.
        for item in chart_runtime:
            chart_res = query_results.get(item["query_key"], None)
            if not chart_res or not chart_res.get("ok"):
                output["charts"].append({
                    "id": item["chart_id"],
                    "title": item["title"],
                    "type": item["type"],
                    "xlabel": item["xlabel"],
                    "ylabel": item["ylabel"],
                    "sql": item["query_key"],
                    "_refresh_failed": True,
                    "showDataLabels": item["showDataLabels"],
                })
                continue

            df = chart_res.get("df")
            plan = {
                "title": item["title"],
                "type": item["type"],
                "xlabel": item["xlabel"],
                "ylabel": item["ylabel"],
            }
            c_data = _build_custom_chart_payload(plan, df)
            _maybe_refine_short_window_line_to_weekly(
                connection=connection,
                chart_sql=item.get("query_key", ""),
                c_data=c_data,
                source_table_base=source_table_base,
                source_table_query=source_table_query,
                where_sql=where_sql,
                date_column=date_column,
                applied_start_date=applied_start_date,
                applied_end_date=applied_end_date,
                logs=log,
                context=f"Filter Chart {item.get('chart_id', '')}",
            )
            c_data["id"] = item["chart_id"]
            c_data["sql"] = c_data.get("sql") or chart_res.get("executed_sql") or ""
            c_data["showDataLabels"] = item["showDataLabels"]
            output["charts"].append(c_data)

        output["__cache"] = {
            "filters": output.get("filters", []),
            "source_table": source_table_base,
            "date_range": output.get("date_range"),
            "selected_date_column": output.get("selected_date_column"),
        }
        return output
    finally:
        connection.close()

def generate_custom_chart_from_prompt_databricks(user_prompt, active_filters_json='{}', clarification_choice=None, allow_ambiguity_fallback=False):
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
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        where_sql, filter_count = _build_databricks_where_clause(
            active_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )
        if filter_count > 0:
            llm_logs.append(f"[FILTER] Applied {filter_count} filter(s) in Databricks mode")

        normalized_choice = _normalize_custom_chart_clarification_choice(clarification_choice, schema_columns)
        ambiguity = None
        if not allow_ambiguity_fallback:
            ambiguity = _detect_custom_chart_ambiguity(
                connection,
                user_prompt,
                schema_columns,
                source_table_query,
                where_sql=where_sql,
                clarification_choice=normalized_choice,
                logs=llm_logs,
            )
        else:
            llm_logs.append("[FALLBACK] Ambiguity fallback enabled for custom chart: letting LLM choose best match")

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

        ai_plan = _enforce_requested_time_grain_on_chart_plan(ai_plan, prompt_for_chart, preferred_date_col=date_column)

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
            ai_plan = _enforce_requested_time_grain_on_chart_plan(ai_plan, prompt_for_chart, preferred_date_col=date_column)
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
        chart_payload["sql"] = executed_sql
        return {
            "chart": chart_payload,
            "generated_sql": executed_sql,
            "tokens_used": tokens_used,
            "logs": llm_logs,
            "data_mode": "databricks",
        }
    finally:
        connection.close()

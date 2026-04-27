import pandas as pd
import json
import os
import re
import time
import math
import numpy as np
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from django.http import HttpResponse
import zipfile
import io
import os
from django.conf import settings
import uuid
from datetime import datetime
try:
    import psycopg2
except Exception:
    psycopg2 = None

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
NON_MAP_CHART_TARGET = 4
TOTAL_CHART_TARGET = NON_MAP_CHART_TARGET + 1
PIE_TOP_LABEL_LIMIT = 15
_kb_cache = {}  # module_name -> (data, fetched_at_timestamp)
KB_CACHE_TTL_SECONDS = 3600  # 1 hour
KB_MODULE_ENV_VAR = "KB_MODULE_NAME"
KB_ENABLED_ENV_VAR = "KB_ENABLED"
KB_ENABLED_DEFAULT = "true"
KB_CONTEXT_MAX_CHARS_DEFAULT = 12000

# Physical-table names the LLM may mention from KB formulas. Runtime SQL must
# execute against analysis_view, which already includes relationship columns.
DATABRICKS_PHYSICAL_TABLE_TAILS = {
    "fact_invoice",
    "dim_product_master",
    "dim_customer_master",
    "fill_rate",
    "final_invoice_with_material",
    "dim_retailer_master",
}

DATABRICKS_DIM_PREFIX_BY_TABLE = {
    "dim_product_master": "product",
    "dim_customer_master": "customer",
    "dim_retailer_master": "retailer",
    "fill_rate": "fill_rate",
    "final_invoice_with_material": "suspicious",
}

# Unqualified business-column names often emitted by LLM outputs that should
# resolve to denormalized analysis_view columns.
ANALYSIS_VIEW_ALIAS_HINTS = {
    "bill_date": "BillDate",
    "invoice_number": "InvoiceNumber",
    "invoice_quantity": "InvoiceQuantity",
    "material_no": "Material_No",
    "retailer_uid": "RetailerUID",
    "brand": "product_brand",
    "subbrand": "product_subbrand",
    "product": "product_product",
    "category": "product_cateogry",
    "cateogry": "product_cateogry",
    "item_desc": "product_item_desc",
    "item_cbu_code": "product_item_cbu_code",
    "flavour": "product_flavour",
    "pack": "product_pack",
    "pricepoint": "product_pricepoint",
    "region_name": "customer_region_name",
    "som_name": "customer_som_name",
    "sold_to_code": "customer_sold_to_code",
    "customer_name": "customer_customer_name",
    "asm_name": "customer_asm_name",
    "tsi_name": "customer_tsi_name",
    "cust_channel": "retailer_cust_channel",
    "ret_uid": "retailer_ret_uid",
    "adjusted_inv_quantity": "fill_rate_adjusted_inv_quantity",
    "adjusted_ord_quantity": "fill_rate_adjusted_ord_quantity",
    "ret_rag_flag": "suspicious_ret_rag_flag",
    "inv_retailer": "suspicious_inv_retailer",
    "inv_awcode": "suspicious_inv_awcode",
    "yearmonth": "suspicious_yearmonth",
}

KB_GUARD_BUILD_ID = "kb_guard_2026_04_02_v2"


def _is_databricks_mode_active():
    return config_is_databricks_mode_active()


def _live_server_logs_enabled():
    raw = str(os.getenv("DATABRICKS_LIVE_SERVER_LOGS", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _live_server_log_max_chars():
    raw = str(os.getenv("DATABRICKS_LIVE_SERVER_LOG_MAX_CHARS", "2000")).strip()
    try:
        return max(200, int(raw))
    except Exception:
        return 2000


class _LiveLogBuffer(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._emit_console = _live_server_logs_enabled()
        self._max_chars = _live_server_log_max_chars()
        self._context = str(kwargs.get("context") or "").strip()

    def append(self, item):
        super().append(item)
        if not self._emit_console:
            return
        try:
            text = str(item or "")
            if len(text) > self._max_chars:
                text = f"{text[:self._max_chars]} ... [TRUNCATED {len(str(item)) - self._max_chars} chars]"
            prefix = f"[LIVE] {self._context} " if self._context else "[LIVE] "
            print(prefix + text, flush=True)
        except Exception:
            pass


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

    # For startup/default views, prefer complete months over an in-progress
    # current month so month-level trends do not understate the latest point.
    try:
        month_end = end_ts + pd.offsets.MonthEnd(0)
        if pd.notna(end_ts) and end_ts < month_end:
            end_ts = (end_ts.replace(day=1) - pd.Timedelta(days=1)).normalize()
    except Exception:
        pass

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
        logs.append(f"[FILTER] Default date window applied: {start_d} to {end_d} (last {_default_dashboard_months()} complete months)")

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

        line = (
            f"[LLM REQUEST] {context} | model={model_name} | json_mode={json_mode} | "
            f"messages={len(messages or [])} | chars={payload_chars}\n{preview}"
        )
        if debug_logs is not None:
            debug_logs.append(line)
        if not isinstance(debug_logs, _LiveLogBuffer):
            print(line, flush=True)
    except Exception as e:
        err = f"[LLM REQUEST] {context} | logging failed: {str(e)}"
        if debug_logs is not None:
            debug_logs.append(err)
        print(err, flush=True)


def _log_llm_response(debug_logs, context, tokens_used, content, max_chars=50000):
    try:
        text = content if content is not None else ""
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False)
        text = _redact_sensitive_text(text)

        payload_chars = len(text)
        preview = text[:max_chars]
        if payload_chars > max_chars:
            preview += f"\n... [TRUNCATED {payload_chars - max_chars} chars]"

        line = f"[LLM RESPONSE] {context} | tokens={tokens_used} | chars={payload_chars}\n{preview}"
        if debug_logs is not None:
            debug_logs.append(line)
        if not isinstance(debug_logs, _LiveLogBuffer):
            print(line, flush=True)
    except Exception as e:
        err = f"[LLM RESPONSE] {context} | logging failed: {str(e)}"
        if debug_logs is not None:
            debug_logs.append(err)
        print(err, flush=True)


def _step_start(logs, step_name, detail=""):
    start_ts = time.perf_counter()
    if logs is not None:
        suffix = f" | {detail}" if detail else ""
        logs.append(f"[STEP] {step_name} START{suffix}")
    return start_ts


def _step_done(logs, step_name, start_ts, detail=""):
    if logs is None:
        return
    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    suffix = f" | {detail}" if detail else ""
    logs.append(f"[STEP] {step_name} DONE | {elapsed_ms:.1f} ms{suffix}")


def _step_fail(logs, step_name, start_ts, error):
    if logs is None:
        return
    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    logs.append(f"[ERROR] {step_name} FAILED | {elapsed_ms:.1f} ms | {str(error)}")


def call_ai_with_retry(messages, json_mode=False, retries=3, debug_logs=None, context="LLM"):
    model_name = "gpt-4o"
    if not client:
        msg = f"[LLM REQUEST] {context} skipped: OPENAI_API_KEY not configured"
        if debug_logs is not None:
            debug_logs.append(msg)
        print(msg, flush=True)
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
            err = f"[LLM ERROR] {context} attempt {attempt + 1}/{retries}: {str(e)}"
            if debug_logs is not None:
                debug_logs.append(err)
            print(err, flush=True)
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


def _detect_region_column(schema_columns, column_names):
    """
    Auto-detect the best region/geography column from schema.
    Priority order: explicit region > zone > state > territory > city > area > district.
    """
    priority_keywords = [
        [
            "region",
            "region_name",
            "customer_region_name",
            "customer_region",
            "customer region name",
            "customer region",
            "region name",
            "customer_som_name",
            "customer_asm_name",
        ],
        ["zone", "zone_name", "sales_zone"],
        ["state", "state_name", "st_nm"],
        ["territory", "territory_name"],
        ["city", "city_name"],
        ["area", "area_name"],
        ["district", "district_name"],
    ]

    col_type_lookup = {str(c).lower(): t for c, t in (schema_columns or [])}
    cols = [str(c).strip() for c in (column_names or []) if str(c).strip()]

    for keyword_group in priority_keywords:
        for keyword in keyword_group:
            for col in cols:
                if str(col).lower() == keyword.lower():
                    return col, str(col_type_lookup.get(str(col).lower(), "STRING"))

    for keyword_group in priority_keywords:
        for keyword in keyword_group:
            for col in cols:
                if keyword.lower() in str(col).lower():
                    return col, str(col_type_lookup.get(str(col).lower(), "STRING"))

    # Third pass: normalized exact match (remove spaces/underscores).
    for keyword_group in priority_keywords:
        for keyword in keyword_group:
            kw_norm = re.sub(r"[\s_]+", "", str(keyword).lower())
            for col in cols:
                col_norm = re.sub(r"[\s_]+", "", str(col).lower())
                if col_norm == kw_norm:
                    return col, str(col_type_lookup.get(str(col).lower(), "STRING"))

    # Fourth pass: normalized partial match.
    for keyword_group in priority_keywords:
        for keyword in keyword_group:
            kw_norm = re.sub(r"[\s_]+", "", str(keyword).lower())
            for col in cols:
                col_norm = re.sub(r"[\s_]+", "", str(col).lower())
                if kw_norm in col_norm or col_norm in kw_norm:
                    return col, str(col_type_lookup.get(str(col).lower(), "STRING"))

    # Fifth pass: strip common join prefixes and retry.
    join_prefixes = ["customer_", "product_", "dim_", "fact_"]
    for keyword_group in priority_keywords:
        for keyword in keyword_group:
            for col in cols:
                col_stripped = str(col).lower()
                for prefix in join_prefixes:
                    if col_stripped.startswith(prefix):
                        col_stripped = col_stripped[len(prefix):]
                        break
                if col_stripped == str(keyword).lower():
                    return col, str(col_type_lookup.get(str(col).lower(), "STRING"))

    return None, None


def _detect_map_metric_column(schema_columns, num_cols, preferred_names=None):
    """
    Auto-detect the best numeric metric column for map aggregation.
    Prefers revenue/sales/amount style fields.
    """
    preferred = preferred_names or [
        "netamount",
        "net_amount",
        "gross_value",
        "grossvalue",
        "revenue",
        "sales",
        "amount",
        "value",
        "total_value",
        "invoiceamount",
        "invoice_amount",
        "billed_amount",
    ]
    _ = schema_columns
    col_lower_map = {str(c).lower(): c for c in (num_cols or [])}

    for pref in preferred:
        if pref.lower() in col_lower_map:
            return col_lower_map[pref.lower()]

    return num_cols[0] if num_cols else None


def _build_map_chart_from_databricks(
    connection,
    source_table_base,
    source_table_query,
    where_sql,
    column_names,
    schema_columns,
    num_cols,
    metric_profiles=None,
    metric_name_hint="",
    metric_formula_hint="",
    kb_data=None,
    schema_context="",
    use_llm_metric_selection=True,
    logs=None,
):
    """
    Build a dynamic india_map chart from live Databricks data.
    """
    map_chart = {
        "id": "chart_0",
        "title": "Region Performance Map",
        "type": "india_map",
        "xlabel": "Region",
        "ylabel": "Value",
        "sql": "",
        "x": [],
        "y": [],
        "z": [],
        "region_col": "",
        "metric_col": "",
        "columns": [],
        "rows": [],
        "labels": [],
        "parents": [],
        "values": [],
        "measure": [],
        "map_meta": {
            "region_col": "",
            "metric_col": "",
            "metric_label": "Value",
            "total": 0.0,
            "regions": [],
            "region_to_states": {},
            "mapping_type": "direct",
        },
    }
    if logs is not None:
        logs.append(
            f"[MAP] Scanning {len(column_names)} columns for region: "
            f"{[c for c in column_names if 'region' in str(c).lower() or 'zone' in str(c).lower() or 'area' in str(c).lower()]}"
        )

    region_col, _region_col_type = _detect_region_column(schema_columns, column_names)
    if not region_col:
        if logs is not None:
            logs.append("[MAP] No region/geography column detected; map will be empty")
        map_chart["title"] = "Region Performance Map (No Region Column Detected)"
        return map_chart

    metric_col = _detect_map_metric_column(schema_columns, num_cols)
    use_count = metric_col is None

    map_chart["region_col"] = region_col
    map_chart["metric_col"] = metric_col or "COUNT(*)"
    map_chart["xlabel"] = region_col.replace("_", " ").title()
    map_chart["ylabel"] = (metric_col or "Count").replace("_", " ").title()

    region_ident = _quote_identifier(region_col)
    agg_expr = "COUNT(*)" if use_count else f"SUM({_quote_identifier(metric_col)})"
    metric_profiles_list = metric_profiles if isinstance(metric_profiles, list) else []
    has_explicit_metric_hint = bool(str(metric_name_hint or "").strip() or str(metric_formula_hint or "").strip())
    selected_metric_profile = None
    if has_explicit_metric_hint:
        selected_metric_profile = _resolve_metric_profile_for_hints(
            metric_profiles_list,
            metric_name=metric_name_hint,
            metric_formula=metric_formula_hint,
            preferred_columns=[metric_col] if metric_col else [],
        )
    if selected_metric_profile and logs is not None and str(metric_name_hint or "").strip():
        logs.append(f"[MAP] Reusing hinted map metric: {str((selected_metric_profile or {}).get('name') or '').strip()}")
    map_default_profile = _resolve_default_map_metric_profile(
        metric_profiles_list,
        metric_col=metric_col,
    )
    if not selected_metric_profile and map_default_profile:
        selected_metric_profile = map_default_profile
        if logs is not None:
            logs.append(f"[MAP] Using default startup map metric: {str((map_default_profile or {}).get('name') or '').strip()}")
    if not selected_metric_profile and use_llm_metric_selection:
        selected_metric_profile = _select_map_metric_profile_with_llm(
            metric_profiles_list,
            region_col=region_col,
            schema_context=schema_context,
            kb_data=kb_data,
            logs=logs,
        )

    metric_formula = str((selected_metric_profile or {}).get("formula") or "").strip()
    metric_expr = _resolve_metric_expression_for_manual_map(
        metric_formula=metric_formula,
        selected_metric_profile=selected_metric_profile,
        y_column=metric_col or "",
        agg_raw="SUM" if not use_count else "COUNT",
        column_type_lookup={str(c).lower(): str(t) for c, t in (schema_columns or [])},
        available_columns=column_names,
    )
    if not metric_expr:
        metric_expr = agg_expr
    if not metric_formula:
        metric_formula = metric_expr
    metric_label = str((selected_metric_profile or {}).get("name") or "").strip()
    if not metric_label:
        metric_label = "Count" if use_count else metric_col.replace("_", " ").title()
    metric_unit = str((selected_metric_profile or {}).get("unit_type") or "").strip().lower()
    if not metric_unit:
        metric_unit = _infer_metric_unit_type(
            metric_name=metric_label,
            formula_text=metric_formula,
            expression_text=metric_expr,
        )
    metric_primary_col = str((selected_metric_profile or {}).get("primary_column") or "").strip() or (metric_col or "")
    metric_agg = str((selected_metric_profile or {}).get("aggregation") or "").strip().upper()
    if metric_agg not in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
        metric_agg = "COUNT" if use_count else "SUM"

    map_sql = (
        f"SELECT CAST({region_ident} AS STRING) AS x, "
        f"{metric_expr} AS y "
        f"FROM {DATABRICKS_LOGICAL_VIEW_NAME} "
        f"WHERE {region_ident} IS NOT NULL "
        f"AND TRIM(CAST({region_ident} AS STRING)) != '' "
        f"GROUP BY 1 "
        f"ORDER BY 2 DESC "
        f"LIMIT 30"
    )

    if logs is not None:
        logs.append(
            f"[MAP] Region col='{region_col}' metric expr='{metric_expr}' "
            f"metric profile='{metric_label or 'auto'}'"
        )

    # Use base source when possible; switch to joined source if detected columns
    # exist only on relationship-enriched schema (e.g. customer_region_name).
    base_col_names = {str(c).strip().lower() for c, _ in (_describe_databricks_table_columns(connection, source_table_base) or [])}
    expr_cols = _extract_metric_formula_columns(metric_expr, column_names)
    needed_cols = {str(region_col).strip().lower()}
    if metric_col:
        needed_cols.add(str(metric_col).strip().lower())
    for c in expr_cols:
        needed_cols.add(str(c).strip().lower())
    needs_join_source = any(c and c not in base_col_names for c in needed_cols)
    map_query_source = source_table_query if needs_join_source else source_table_base
    if logs is not None:
        logs.append(
            f"[MAP] Query source={'joined' if needs_join_source else 'base'} "
            f"(region_in_base={'yes' if str(region_col).strip().lower() in base_col_names else 'no'})"
        )

    try:
        map_df, map_sql_exec = _execute_databricks_user_sql(
            connection,
            map_sql,
            source_table_base,
            query_source=map_query_source,
            where_sql=where_sql,
            available_columns=column_names,
            logs=logs,
            context="India Map Chart",
        )

        if map_df.empty:
            if logs is not None:
                logs.append("[MAP] Map query returned no rows")
            map_chart["title"] = "Region Performance Map (No Data)"
            return map_chart

        map_df.columns = [str(c).lower() for c in map_df.columns]
        x_vals = map_df["x"].astype(str).tolist() if "x" in map_df else map_df.iloc[:, 0].astype(str).tolist()
        y_vals = map_df["y"].fillna(0).astype(float).tolist() if "y" in map_df else map_df.iloc[:, 1].fillna(0).astype(float).tolist()

        total = float(sum(abs(v) for v in y_vals))
        regions_meta = []
        for i in range(len(x_vals)):
            val = float(y_vals[i]) if i < len(y_vals) else 0.0
            pct = round((val / total * 100.0), 2) if total > 0 else 0.0
            regions_meta.append({
                "name": str(x_vals[i]),
                "value": val,
                "pct": pct,
            })

        # Default business-region to India-state lookup for map rendering when
        # region values are business clusters (North 1, South 2, etc.).
        BUSINESS_REGION_TO_STATES = {
            # North — all northern states combined, shared by both North 1 and North 2
            "north 1": [
                "Jammu and Kashmir", "Ladakh", "Himachal Pradesh",
                "Punjab", "Uttaranchal", "Uttarakhand", "Chandigarh",
                "Haryana", "Delhi", "Uttar Pradesh", "Rajasthan",
            ],
            "north 2": [
                "Jammu and Kashmir", "Ladakh", "Himachal Pradesh",
                "Punjab", "Uttaranchal", "Uttarakhand", "Chandigarh",
                "Haryana", "Delhi", "Uttar Pradesh", "Rajasthan",
            ],
            "north1": [
                "Jammu and Kashmir", "Ladakh", "Himachal Pradesh",
                "Punjab", "Uttaranchal", "Uttarakhand", "Chandigarh",
                "Haryana", "Delhi", "Uttar Pradesh", "Rajasthan",
            ],
            "north2": [
                "Jammu and Kashmir", "Ladakh", "Himachal Pradesh",
                "Punjab", "Uttaranchal", "Uttarakhand", "Chandigarh",
                "Haryana", "Delhi", "Uttar Pradesh", "Rajasthan",
            ],
            "north": [
                "Jammu and Kashmir", "Ladakh", "Himachal Pradesh",
                "Punjab", "Uttaranchal", "Uttarakhand", "Chandigarh",
                "Haryana", "Delhi", "Uttar Pradesh", "Rajasthan",
            ],
            # South — all southern states combined, shared by both South 1 and South 2
            "south 1": [
                "Andhra Pradesh", "Telangana", "Karnataka",
                "Tamil Nadu", "Kerala", "Puducherry", "Lakshadweep",
            ],
            "south 2": [
                "Andhra Pradesh", "Telangana", "Karnataka",
                "Tamil Nadu", "Kerala", "Puducherry", "Lakshadweep",
            ],
            "south1": [
                "Andhra Pradesh", "Telangana", "Karnataka",
                "Tamil Nadu", "Kerala", "Puducherry", "Lakshadweep",
            ],
            "south2": [
                "Andhra Pradesh", "Telangana", "Karnataka",
                "Tamil Nadu", "Kerala", "Puducherry", "Lakshadweep",
            ],
            "south": [
                "Andhra Pradesh", "Telangana", "Karnataka",
                "Tamil Nadu", "Kerala", "Puducherry", "Lakshadweep",
            ],
            # East
            "east": [
                "Bihar", "Jharkhand", "Orissa", "Odisha", "West Bengal",
                "Assam", "Meghalaya", "Manipur", "Tripura",
                "Nagaland", "Arunachal Pradesh", "Mizoram", "Sikkim",
            ],
            # West
            "west": [
                "Gujarat", "Maharashtra", "Goa",
                "Dadra and Nagar Haveli", "Daman and Diu",
                "Dadra and Nagar Haveli and Daman and Diu",
            ],
            # Central
            "central": [
                "Madhya Pradesh", "Chhattisgarh",
            ],
        }

        map_chart["x"] = x_vals
        map_chart["y"] = y_vals
        map_chart["sql"] = map_sql_exec
        map_chart["title"] = f"{metric_label} Map"
        map_chart["ylabel"] = metric_label
        map_chart["metric_col"] = metric_expr
        map_chart["map_meta"] = {
            "region_col": region_col,
            "metric_col": metric_expr,
            "dimension_col": region_col,
            "metric_label": metric_label,
            "total": total,
            "regions": regions_meta,
            "region_to_states": BUSINESS_REGION_TO_STATES,
            "mapping_type": "business_region",
        }
        map_chart["manual_config"] = {
            "x_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
            "x_column": region_col,
            "y_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
            "y_column": metric_primary_col,
            "aggregation": metric_agg,
            "chart_type": "india_map",
            "metric_name": metric_label,
            "metric_formula": metric_formula,
            "metric_unit": metric_unit,
            "metric_axis": "y",
            "dimension_axis": "x",
        }

        if logs is not None:
            logs.append(
                f"[MAP] Loaded map data: title='{map_chart['title']}' regions={len(x_vals)} total={total:,.0f} "
                f"top={x_vals[0] if x_vals else 'N/A'}"
            )
    except Exception as e:
        if logs is not None:
            logs.append(f"[WARN] Map chart data fetch failed: {str(e)}")
        map_chart["title"] = "Region Performance Map (Load Error)"

    return map_chart



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
    label_l = str(label_text or "").strip().lower()
    sql_l = str(sql_text or "").strip().lower()
    prompt_l = str(prompt_text or "").strip().lower()
    text = f"{label_l} {sql_l} {prompt_l}"

    has_invoice = any(tok in text for tok in ["invoice", "invoices", "bill", "bills"])
    if not has_invoice:
        return False

    # Do not rewrite ratio/value KPIs that happen to include COUNT(...) in denominator.
    non_count_markers = [
        "abv", "average bill", "avg bill", "daily run rate", "drr",
        "sales", "revenue", "value", "ulpo", "fill rate", "suspicious eco",
        "ratio", "percent", "%"
    ]
    if any(m in label_l for m in non_count_markers):
        return False
    if any(m in prompt_l for m in non_count_markers):
        return False
    if "/" in sql_l and "count(" in sql_l and any(fn in sql_l for fn in ["sum(", "avg(", "min(", "max("]):
        return False

    count_like_tokens = [
        "count",
        "distinct",
        "number",
        "how many",
        "no of",
        "no.",
    ]
    label_count_like = any(tok in label_l for tok in count_like_tokens)
    prompt_count_like = any(tok in prompt_l for tok in count_like_tokens)
    if label_count_like or prompt_count_like:
        return True

    # SQL fallback: pure invoice-count shape only.
    sql_no_ws = re.sub(r"\s+", " ", sql_l)
    if "count(" in sql_no_ws and not any(fn in sql_no_ws for fn in ["sum(", "avg(", "min(", "max("]):
        if "/" not in sql_no_ws and any(tok in sql_no_ws for tok in ["invoice", "bill"]):
            return True
    return False


def _normalize_invoice_count_kpi_plan(kpi_obj, schema_columns, table_name, date_column=None, prompt_text="", logs=None, entity_col=None):
    if not isinstance(kpi_obj, dict):
        return None

    out = dict(kpi_obj)
    label_text = str(out.get("label", "")).strip()
    sql_text = str(out.get("sql") or out.get("value_sql") or "").strip()
    metric_name = str(out.get("metric_name") or "").strip().lower()

    if metric_name in {"abv", "drr", "sales", "ulpo", "fill rate", "suspicious eco", "units"}:
        return out

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


def _sql_logging_enabled():
    return _safe_bool_env("DATABRICKS_LOG_SQL_TEXT", False)


def _sql_log_max_chars():
    raw = os.getenv("DATABRICKS_LOG_SQL_TEXT_MAX_CHARS", "25000")
    try:
        return max(200, int(raw))
    except Exception:
        return 25000


def _log_sql_text(debug_logs, context, label, sql_text):
    if debug_logs is None or not _sql_logging_enabled():
        return
    try:
        text = _redact_sensitive_text(sql_text if sql_text is not None else "")
        text = str(text)
        total_chars = len(text)
        max_chars = _sql_log_max_chars()
        preview = text[:max_chars]
        if total_chars > max_chars:
            preview += f"\n... [TRUNCATED {total_chars - max_chars} chars]"
        debug_logs.append(
            f"[SQL] {context} | {label} | chars={total_chars}\n{preview}"
        )
    except Exception as e:
        debug_logs.append(f"[SQL] {context} | {label} | logging failed: {str(e)}")


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


def _dedupe_schema_columns(columns):
    deduped = []
    seen = set()
    dropped = 0
    for col_name, col_type in (columns or []):
        name = str(col_name or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append((col_name, col_type))
    return deduped, dropped


def _derive_dim_prefix_from_table_name(table_name):
    table_tail = _extract_table_tail(table_name)
    if not table_tail:
        return "dim"

    mapped = DATABRICKS_DIM_PREFIX_BY_TABLE.get(table_tail)
    if mapped:
        return str(mapped).strip().lower()

    normalized = clean_col_name(table_tail)
    if normalized.startswith("dim_"):
        normalized = normalized[4:]
    if normalized.startswith("fact_"):
        normalized = normalized[5:]
    if normalized.endswith("_master"):
        normalized = normalized[:-7]
    normalized = normalized.strip("_")
    return normalized or "dim"


def _build_kb_selected_columns_lookup(kb_data):
    if not isinstance(kb_data, dict):
        return {}

    selected_map = _normalize_selected_columns_map(kb_data.get("selected_columns"))
    out = {}
    for table_key, payload in (selected_map or {}).items():
        table_name = str((payload or {}).get("table_name") or table_key).strip()
        table_tail = _extract_table_tail(table_name) or table_name.lower()
        cols = set()
        for col in (payload or {}).get("columns") or []:
            col_name = str((col or {}).get("name") or "").strip()
            if col_name:
                cols.add(col_name.lower())
        out[table_tail.lower()] = cols
    return out


def _identifier_tokens(value):
    text = str(value or "").strip().lower()
    if not text:
        return set()
    parts = re.split(r"[^a-z0-9]+", text)
    expanded = []
    for p in parts:
        if not p:
            continue
        expanded.append(p)
        if p.startswith("inv") and len(p) > 3:
            expanded.append(p[3:])
        if p.endswith("uid"):
            expanded.append("retailer")
            expanded.append("id")
        if p in {"billdate", "newbilldate"}:
            expanded.extend(["bill", "date"])
        if p == "yearmonth":
            expanded.extend(["year", "month", "date"])
    return {x for x in expanded if x}


def _identifier_semantic_family(value):
    raw = clean_col_name(value)
    tokens = _identifier_tokens(value)
    if not raw:
        return ""

    if "awcode" in raw or raw.startswith("aw_") or raw.endswith("_aw") or raw == "aw_code" or "inv_awcode" in raw:
        return "aw_code"

    if "yearmonth" in raw or "yearmon" in raw or ("year" in tokens and "month" in tokens):
        return "month_period"

    if raw in {"billdate", "newbilldate"} or ("bill" in tokens and "date" in tokens):
        return "exact_date"

    if (
        "retaileruid" in raw
        or "ret_uid" in raw
        or "inv_retailer" in raw
        or (("retailer" in tokens or "ret" in tokens) and ("uid" in tokens or "id" in tokens))
    ):
        return "retailer_id"

    if "material" in raw and ("no" in tokens or "number" in tokens or "material" in tokens):
        return "material_id"

    if "invoice" in raw and ("number" in tokens or "no" in tokens):
        return "invoice_id"

    return ""


def _identifier_pair_score(left_name, right_name):
    lt = _identifier_tokens(left_name)
    rt = _identifier_tokens(right_name)
    if not lt or not rt:
        return 0.0
    overlap = len(lt.intersection(rt))
    score = float(overlap * 3)
    left_s = str(left_name or "").lower()
    right_s = str(right_name or "").lower()
    left_family = _identifier_semantic_family(left_name)
    right_family = _identifier_semantic_family(right_name)
    if left_family and right_family:
        if left_family == right_family:
            score += 8.0
        else:
            score -= 8.0
    if ("uid" in left_s or "retailer" in left_s) and ("uid" in right_s or "retailer" in right_s):
        score += 4.0
    if ("yearmon" in left_s or "yearmonth" in left_s) and ("yearmon" in right_s or "yearmonth" in right_s):
        score += 6.0
    if "bill" in left_s and "date" in right_s and "month" not in right_s:
        score += 3.0
    if "aw" in left_s and "aw" in right_s:
        score += 3.0
    if "material" in left_s and "material" in right_s:
        score += 3.0
    return score


def _normalized_join_operand_sql(table_alias, column_name, semantic_family=""):
    alias = str(table_alias or "").strip()
    col = _quote_identifier(column_name)
    family = str(semantic_family or "").strip().lower()
    base_expr = f"TRIM(UPPER(CAST({alias}.{col} AS STRING)))"
    if family == "month_period":
        # Normalize month keys to digits-only YYYYMM form with low-cost replacements
        # (faster than regex on large joins).
        base_trim = f"TRIM(CAST({alias}.{col} AS STRING))"
        return (
            "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
            f"{base_trim}, '-', ''), '/', ''), ' ', ''), '.', ''), '_', '')"
        )
    return base_expr


def _build_join_predicate_sql(base_alias, base_key, dim_alias, dim_key):
    left_family = _identifier_semantic_family(base_key)
    right_family = _identifier_semantic_family(dim_key)
    left_expr = _normalized_join_operand_sql(base_alias, base_key, left_family)
    right_expr = _normalized_join_operand_sql(dim_alias, dim_key, right_family)
    if left_family == "month_period" and right_family == "month_period":
        return (
            f"{left_expr} = {right_expr} "
            f"AND LENGTH({left_expr}) = 6 "
            f"AND LENGTH({right_expr}) = 6"
        )
    return f"{left_expr} = {right_expr}"


def _repair_join_key_semantics(join_keys, available_left_columns=None, available_right_columns=None):
    pairs = [
        (str(a or "").strip(), str(b or "").strip())
        for a, b in (join_keys or [])
        if str(a or "").strip() and str(b or "").strip()
    ]
    if not pairs:
        return pairs

    left_candidates = [str(c).strip() for c in (available_left_columns or []) if str(c).strip()]
    right_candidates = [str(c).strip() for c in (available_right_columns or []) if str(c).strip()]
    if not left_candidates and not right_candidates:
        return pairs

    used_left = {a.lower() for a, _ in pairs if a}
    used_right = {b.lower() for _, b in pairs if b}
    repaired = []

    for left_name, right_name in pairs:
        cur_left = str(left_name or "").strip()
        cur_right = str(right_name or "").strip()
        left_family = _identifier_semantic_family(cur_left)
        right_family = _identifier_semantic_family(cur_right)

        if left_family and right_family and left_family != right_family:
            best_left = cur_left
            best_right = cur_right
            best_score = _identifier_pair_score(cur_left, cur_right)

            if right_family and left_candidates:
                for cand_left in left_candidates:
                    cand_left_s = str(cand_left).strip()
                    if not cand_left_s:
                        continue
                    cand_family = _identifier_semantic_family(cand_left_s)
                    if cand_family != right_family:
                        continue
                    cand_score = _identifier_pair_score(cand_left_s, cur_right)
                    if cand_left_s.lower() != cur_left.lower() and cand_left_s.lower() in used_left:
                        cand_score -= 4.0
                    if cand_score > best_score:
                        best_left = cand_left_s
                        best_score = cand_score

            left_family_for_right = _identifier_semantic_family(best_left)
            if left_family_for_right and right_candidates:
                for cand_right in right_candidates:
                    cand_right_s = str(cand_right).strip()
                    if not cand_right_s:
                        continue
                    cand_family = _identifier_semantic_family(cand_right_s)
                    if cand_family != left_family_for_right:
                        continue
                    cand_score = _identifier_pair_score(best_left, cand_right_s)
                    if cand_right_s.lower() != cur_right.lower() and cand_right_s.lower() in used_right:
                        cand_score -= 4.0
                    if cand_score > best_score:
                        best_right = cand_right_s
                        best_score = cand_score

            if best_left.lower() != cur_left.lower():
                used_left.discard(cur_left.lower())
                used_left.add(best_left.lower())
                cur_left = best_left
            if best_right.lower() != cur_right.lower():
                used_right.discard(cur_right.lower())
                used_right.add(best_right.lower())
                cur_right = best_right

        repaired.append((cur_left, cur_right))

    return repaired


def _optimize_join_key_pairing(oriented_keys):
    pairs = [(str(a or "").strip(), str(b or "").strip()) for a, b in (oriented_keys or []) if str(a or "").strip() and str(b or "").strip()]
    if len(pairs) < 2:
        return pairs

    left_cols = [a for a, _ in pairs]
    right_cols = [b for _, b in pairs]
    if len(set(left_cols)) != len(left_cols) or len(set(right_cols)) != len(right_cols):
        return pairs
    if len(left_cols) > 6:
        return pairs

    import itertools
    current_score = sum(_identifier_pair_score(a, b) for a, b in pairs)
    best_pairs = pairs
    best_score = current_score
    for perm in itertools.permutations(right_cols):
        candidate = list(zip(left_cols, perm))
        cand_score = sum(_identifier_pair_score(a, b) for a, b in candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_pairs = candidate

    # Only remap when improvement is meaningful; otherwise preserve KB order.
    if best_score >= current_score + 3.0:
        return best_pairs
    return pairs


def _build_kb_relation_specs(base_table, kb_data, logs=None):
    if not isinstance(kb_data, dict):
        if logs is not None:
            logs.append("[SOURCE] KB relation specs unavailable: kb_data missing")
        return []

    base_tail = _extract_table_tail(base_table)
    if not base_tail:
        if logs is not None:
            logs.append("[SOURCE] KB relation specs unavailable: could not resolve base table tail")
        return []

    groups = _extract_relationship_join_groups(kb_data.get("relationships"))
    selected_lookup = _build_kb_selected_columns_lookup(kb_data)
    left_selected_cols = list(selected_lookup.get(base_tail.lower(), set())) if isinstance(selected_lookup, dict) else []
    spec_by_table = {}

    for group in (groups or []):
        left_table = str(group.get("left_table") or "").strip()
        right_table = str(group.get("right_table") or "").strip()
        rel_keys = list(group.get("keys") or [])
        if not left_table or not right_table or not rel_keys:
            continue

        left_tail = _extract_table_tail(left_table) or left_table.lower()
        right_tail = _extract_table_tail(right_table) or right_table.lower()

        target_table = ""
        oriented_keys = []
        if left_tail == base_tail and right_tail != base_tail:
            target_table = right_table
            oriented_keys = [(str(lc).strip(), str(rc).strip()) for lc, rc in rel_keys]
        elif right_tail == base_tail and left_tail != base_tail:
            target_table = left_table
            oriented_keys = [(str(rc).strip(), str(lc).strip()) for lc, rc in rel_keys]
        else:
            continue

        target_tail = _extract_table_tail(target_table) or target_table.lower()
        if not target_tail:
            continue

        oriented_keys = _optimize_join_key_pairing(oriented_keys)
        target_selected_cols = list(selected_lookup.get(target_tail.lower(), set())) if isinstance(selected_lookup, dict) else []
        oriented_keys = _repair_join_key_semantics(
            oriented_keys,
            available_left_columns=left_selected_cols,
            available_right_columns=target_selected_cols,
        )

        prefix = _derive_dim_prefix_from_table_name(target_table)
        DATABRICKS_DIM_PREFIX_BY_TABLE[target_tail] = prefix

        if target_tail not in spec_by_table:
            spec_by_table[target_tail] = {
                "table": target_table,
                "join_keys": [],
                "prefix": prefix,
                "include_by_default": True,
                "selected_columns": set(selected_lookup.get(target_tail, set())),
                "seen_pairs": set(),
            }

        for base_col, dim_col in oriented_keys:
            if not base_col or not dim_col:
                continue
            pair_sig = f"{base_col.lower()}::{dim_col.lower()}"
            if pair_sig in spec_by_table[target_tail]["seen_pairs"]:
                continue
            spec_by_table[target_tail]["seen_pairs"].add(pair_sig)
            spec_by_table[target_tail]["join_keys"].append((base_col, dim_col))

    specs = []
    for target_tail in sorted(spec_by_table.keys()):
        spec = spec_by_table[target_tail]
        if not spec.get("join_keys"):
            continue
        spec["selected_columns"] = sorted(spec.get("selected_columns") or [])
        spec.pop("seen_pairs", None)
        specs.append(spec)

    if logs is not None:
        logs.append(f"[SOURCE] KB relation specs resolved: {len(specs)} table(s)")
    return specs


def _kb_joinable_table_tails(base_table, kb_data):
    specs = _build_kb_relation_specs(base_table, kb_data, logs=None)
    tails = set()
    base_tail = _extract_table_tail(base_table)
    if base_tail:
        tails.add(base_tail)
    for spec in specs:
        tail = _extract_table_tail(spec.get("table"))
        if tail:
            tails.add(tail)
    return tails


def _build_databricks_relationship_select_model(connection, base_table, base_columns, required_columns=None, logs=None, kb_data=None):
    relation_specs = _build_kb_relation_specs(base_table, kb_data, logs=logs)
    if not relation_specs:
        if logs is not None:
            logs.append("[SOURCE] KB relation specs empty; skipping relation-source build")
        return None

    base_col_names = [c for c, _ in (base_columns or [])]
    if not base_col_names:
        return None

    required_lookup = None
    if required_columns:
        required_lookup = {str(c).strip().lower() for c in required_columns if str(c).strip()}

    def _join_debug_enabled():
        env_raw = str(os.getenv("DATABRICKS_JOIN_DEBUG", "false")).strip().lower()
        return env_raw in {"1", "true", "yes", "on"}

    run_join_debug = _join_debug_enabled()

    select_clauses = []
    schema_columns = []
    used_aliases = set()

    for col_name, _ in base_columns:
        used_aliases.add(str(col_name).lower())

    for col_name, col_type in base_columns:
        if required_lookup is not None and str(col_name).lower() not in required_lookup:
            continue
        select_clauses.append(f"f.{_quote_identifier(col_name)} AS {_quote_identifier(col_name)}")
        schema_columns.append((col_name, col_type))

    join_clauses = []
    joined_tables = []
    alias_idx = 1

    for spec in relation_specs:
        if required_lookup is None and not spec.get("include_by_default", True):
            continue

        dim_fqn = _resolve_related_databricks_table_fqn(base_table, spec["table"])
        if not dim_fqn:
            continue

        try:
            dim_columns_raw = _describe_databricks_table_columns(connection, dim_fqn)
        except Exception as e:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: describe failed ({str(e)})")
            continue

        dim_columns, dropped_dim_cols = _dedupe_schema_columns(dim_columns_raw)
        if dropped_dim_cols and logs is not None:
            logs.append(f"[SOURCE] Deduped {dropped_dim_cols} duplicate columns in {spec['table']}")

        if not dim_columns:
            if logs is not None:
                logs.append(f"[SOURCE] Skip relation join {spec['table']}: no columns returned")
            continue

        dim_col_names = [c for c, _ in dim_columns]
        resolved_join_keys = []
        missing_pairs = []
        for base_candidate, dim_candidate in spec.get("join_keys") or []:
            base_key = _find_col_case_insensitive(base_col_names, [base_candidate])
            dim_key = _find_col_case_insensitive(dim_col_names, [dim_candidate])
            if not base_key or not dim_key:
                missing_pairs.append(f"{base_candidate}->{dim_candidate}")
                continue
            resolved_join_keys.append((base_key, dim_key))

        resolved_join_keys = _repair_join_key_semantics(
            resolved_join_keys,
            available_left_columns=base_col_names,
            available_right_columns=dim_col_names,
        )
        if logs is not None and resolved_join_keys:
            original_sig = " | ".join(f"{a}->{b}" for a, b in (spec.get("join_keys") or []))
            repaired_sig = " | ".join(f"{a}->{b}" for a, b in resolved_join_keys)
            if original_sig != repaired_sig:
                logs.append(f"[JOIN FIX] Semantic join repair for {spec['table']}: {original_sig} => {repaired_sig}")

        if not resolved_join_keys:
            if logs is not None:
                msg = ", ".join(missing_pairs) if missing_pairs else "none"
                logs.append(
                    f"[SOURCE] Skip relation join {spec['table']}: join keys not found ({msg})"
                )
            continue

        dim_alias = f"d{alias_idx}"
        alias_idx += 1

        dim_select_clauses = []
        dim_schema_columns = []
        allowed_dim_cols = {str(c).strip().lower() for c in (spec.get("selected_columns") or []) if str(c).strip()}
        for dim_col_name, dim_col_type in dim_columns:
            if allowed_dim_cols and str(dim_col_name).strip().lower() not in allowed_dim_cols:
                continue
            cleaned = clean_col_name(dim_col_name)
            alias_root = f"{spec['prefix']}_{cleaned}" if cleaned else f"{spec['prefix']}_col"
            alias_name = alias_root
            suffix = 2
            while alias_name.lower() in used_aliases:
                alias_name = f"{alias_root}_{suffix}"
                suffix += 1

            if required_lookup is not None and alias_name.lower() not in required_lookup:
                continue

            used_aliases.add(alias_name.lower())
            dim_select_clauses.append(
                f"{dim_alias}.{_quote_identifier(dim_col_name)} AS {_quote_identifier(alias_name)}"
            )
            dim_schema_columns.append((alias_name, dim_col_type))

        if required_lookup is not None and not dim_select_clauses:
            continue

        join_predicates = []
        for base_key, dim_key in resolved_join_keys:
            join_predicates.append(_build_join_predicate_sql("f", base_key, dim_alias, dim_key))

        if run_join_debug and logs is not None and join_predicates:
            try:
                coverage_sql = (
                    f"SELECT COUNT(*) AS total_rows, "
                    f"COUNT({dim_alias}.{_quote_identifier(resolved_join_keys[0][1])}) AS matched_rows "
                    f"FROM {base_table} f "
                    f"LEFT JOIN {dim_fqn} {dim_alias} ON {' AND '.join(join_predicates)}"
                )
                coverage_df = fetch_dataframe(connection, coverage_sql, readonly=True)
                if coverage_df is not None and not coverage_df.empty:
                    total_rows = int(coverage_df.iloc[0].get("total_rows") or 0)
                    matched_rows = int(coverage_df.iloc[0].get("matched_rows") or 0)
                    match_pct = (matched_rows / total_rows * 100.0) if total_rows > 0 else 0.0
                    logs.append(
                        f"[JOIN DEBUG] {spec['table']} coverage: total_rows={total_rows}, "
                        f"matched_rows={matched_rows}, match_pct={match_pct:.2f}%"
                    )
                    if total_rows > 0 and match_pct < 60.0:
                        logs.append(
                            f"[JOIN GUARD] Low join coverage for {spec['table']} "
                            f"({match_pct:.2f}%). Check key-format compatibility."
                        )
                    try:
                        inner_sql = (
                            f"SELECT COUNT(*) AS inner_rows "
                            f"FROM {base_table} f "
                            f"INNER JOIN {dim_fqn} {dim_alias} ON {' AND '.join(join_predicates)}"
                        )
                        inner_df = fetch_dataframe(connection, inner_sql, readonly=True)
                        inner_rows = int(inner_df.iloc[0].get("inner_rows") or 0) if inner_df is not None and not inner_df.empty else 0
                        if total_rows > 0:
                            delta_pct = abs(total_rows - inner_rows) / total_rows * 100.0
                            logs.append(
                                f"[JOIN DEBUG] {spec['table']} inner-join check: inner_rows={inner_rows}, "
                                f"delta_vs_left={delta_pct:.2f}%"
                            )
                            if delta_pct > 40.0:
                                logs.append(
                                    f"[JOIN GUARD] INNER JOIN differs sharply for {spec['table']} "
                                    f"(delta {delta_pct:.2f}%). Join mismatch likely."
                                )
                    except Exception as e_inner:
                        logs.append(f"[JOIN DEBUG] Inner-join check skipped for {spec['table']}: {str(e_inner)}")
            except Exception as e:
                logs.append(f"[JOIN DEBUG] Coverage check skipped for {spec['table']}: {str(e)}")

        join_clauses.append(
            "LEFT JOIN {dim_table} {dim_alias} ON {predicate}".format(
                dim_table=dim_fqn,
                dim_alias=dim_alias,
                predicate=" AND ".join(join_predicates),
            )
        )
        joined_tables.append(dim_fqn)
        select_clauses.extend(dim_select_clauses)
        schema_columns.extend(dim_schema_columns)

    if not join_clauses:
        return None

    if not select_clauses:
        return None

    joined_query_source = (
        "(SELECT "
        + ", ".join(select_clauses)
        + f" FROM {base_table} f "
        + " ".join(join_clauses)
        + ") __relation_source"
    )

    if run_join_debug and logs is not None:
        schema_names = {str(c).strip().lower() for c, _ in (schema_columns or [])}
        if "billdate" in schema_names and "suspicious_ret_rag_flag" in schema_names:
            try:
                null_diag_sql = (
                    "SELECT DATE_TRUNC('month', CAST(`BillDate` AS DATE)) AS month_bucket, "
                    "COUNT(*) AS total, "
                    "COUNT(`suspicious_ret_rag_flag`) AS matched_flags "
                    f"FROM {joined_query_source} "
                    "WHERE `BillDate` IS NOT NULL "
                    "GROUP BY 1 ORDER BY 1"
                )
                null_diag_df = fetch_dataframe(connection, null_diag_sql, readonly=True)
                if null_diag_df is not None and not null_diag_df.empty:
                    min_cov = 100.0
                    rows = []
                    for _, row in null_diag_df.head(12).iterrows():
                        total = int(row.get("total") or 0)
                        matched = int(row.get("matched_flags") or 0)
                        cov = (matched / total * 100.0) if total > 0 else 0.0
                        min_cov = min(min_cov, cov)
                        rows.append(f"{row.get('month_bucket')}: {matched}/{total} ({cov:.1f}%)")
                    logs.append("[JOIN DEBUG] suspicious flag coverage by month: " + " | ".join(rows))
                    if min_cov < 30.0:
                        logs.append(
                            "[JOIN GUARD] suspicious_ret_rag_flag coverage is low in one or more months; "
                            "Suspicious ECO may appear as 0 due to unmatched join keys."
                        )
            except Exception as e:
                logs.append(f"[JOIN DEBUG] suspicious flag diagnostic skipped: {str(e)}")

    return {
        "query_source": joined_query_source,
        "schema_columns": schema_columns,
        "joined_tables": joined_tables,
    }


def _build_databricks_relationship_virtual_source(connection, base_table, base_columns, include_sample_rows, logs=None, required_columns=None, kb_data=None):
    relation_model = _build_databricks_relationship_select_model(
        connection,
        base_table,
        base_columns,
        required_columns=required_columns,
        logs=logs,
        kb_data=kb_data,
    )
    if not relation_model:
        return None

    schema_context = _load_databricks_schema_context_from_query_source(
        connection,
        relation_model["query_source"],
        include_sample_rows,
        relation_model["schema_columns"],
    )

    if logs is not None:
        logs.append(
            f"[SOURCE] Databricks relationship joins active: base={base_table}, joined={len(relation_model['joined_tables'])}"
        )

    return {
        "base_table": base_table,
        "base_columns": base_columns,
        "query_source": relation_model["query_source"],
        "schema_columns": relation_model["schema_columns"],
        "schema_context": schema_context,
        "joined_tables": relation_model["joined_tables"],
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


def _log_kb_message(logs, message):
    if logs is not None:
        try:
            logs.append(message)
            return
        except Exception:
            pass
    try:
        print(message, flush=True)
    except Exception:
        pass


def _is_kb_enabled():
    raw = str(os.getenv(KB_ENABLED_ENV_VAR, KB_ENABLED_DEFAULT) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_kb_module_name(module_name=None):
    env_module = str(os.getenv(KB_MODULE_ENV_VAR) or "").strip()
    return env_module


def _parse_kb_json_field(field_name, raw_value):
    default_value = [] if field_name in {"rca_list", "tables", "pos_tagging"} else {}
    if raw_value is None:
        return default_value
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return default_value
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
        raise ValueError(f"Invalid JSON type for {field_name}: {type(parsed).__name__}")
    raise TypeError(f"Unsupported JSON value type for {field_name}: {type(raw_value).__name__}")


def _count_numbered_rules(text):
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    numbered = [ln for ln in lines if re.match(r"^\d+[\)\.\:\-]?\s+", ln)]
    if numbered:
        return len(numbered)
    return 1 if lines else 0


def _extract_table_names(tables_data):
    names = []
    seen = set()

    def _add(name):
        table_name = str(name or "").strip()
        if not table_name:
            return
        key = table_name.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(table_name)

    if isinstance(tables_data, list):
        for item in tables_data:
            if isinstance(item, dict):
                _add(item.get("table_name") or item.get("name") or item.get("table"))
            else:
                _add(item)
    elif isinstance(tables_data, dict):
        nested_tables = tables_data.get("tables")
        if isinstance(nested_tables, list):
            for item in nested_tables:
                if isinstance(item, dict):
                    _add(item.get("table_name") or item.get("name") or item.get("table"))
                else:
                    _add(item)
        else:
            for key in tables_data.keys():
                _add(key)

    return names


def _normalize_selected_columns_map(selected_columns):
    out = {}
    seen = {}

    def _add(table_name, column_name, datatype="", description=""):
        t_name = str(table_name or "").strip()
        c_name = str(column_name or "").strip()
        if not t_name or not c_name:
            return
        table_key = t_name.lower()
        col_key = c_name.lower()
        if table_key not in out:
            out[table_key] = {"table_name": t_name, "columns": []}
            seen[table_key] = set()
        if col_key in seen[table_key]:
            return
        seen[table_key].add(col_key)
        out[table_key]["columns"].append(
            {
                "name": c_name,
                "datatype": str(datatype or "").strip(),
                "description": str(description or "").strip(),
            }
        )

    def _extract_col_name(col_item):
        if isinstance(col_item, str):
            return str(col_item).strip(), "", ""
        if isinstance(col_item, dict):
            col_name = (
                col_item.get("name")
                or col_item.get("column")
                or col_item.get("column_name")
                or col_item.get("field")
            )
            datatype = (
                col_item.get("datatype")
                or col_item.get("data_type")
                or col_item.get("type")
            )
            description = (
                col_item.get("description")
                or col_item.get("desc")
            )
            return str(col_name or "").strip(), str(datatype or "").strip(), str(description or "").strip()
        return "", "", ""

    if isinstance(selected_columns, dict):
        for table_name, table_value in selected_columns.items():
            if isinstance(table_value, list):
                for col_item in table_value:
                    col_name, datatype, description = _extract_col_name(col_item)
                    _add(table_name, col_name, datatype=datatype, description=description)
            elif isinstance(table_value, dict):
                nested_cols = table_value.get("columns")
                if isinstance(nested_cols, list):
                    for col_item in nested_cols:
                        col_name, datatype, description = _extract_col_name(col_item)
                        _add(table_name, col_name, datatype=datatype, description=description)
                elif isinstance(nested_cols, dict):
                    for col_name, col_meta in nested_cols.items():
                        if isinstance(col_meta, dict):
                            _add(
                                table_name,
                                col_name,
                                datatype=col_meta.get("datatype") or col_meta.get("data_type") or col_meta.get("type"),
                                description=col_meta.get("description") or col_meta.get("desc"),
                            )
                        else:
                            _add(table_name, col_name)
                else:
                    for col_name in table_value.keys():
                        _add(table_name, col_name)
            else:
                _add(table_name, table_value)
    elif isinstance(selected_columns, list):
        for table_item in selected_columns:
            if not isinstance(table_item, dict):
                continue
            table_name = table_item.get("table_name") or table_item.get("table") or table_item.get("name")
            table_cols = table_item.get("columns")
            if not isinstance(table_cols, list):
                continue
            for col_item in table_cols:
                col_name, datatype, description = _extract_col_name(col_item)
                _add(table_name, col_name, datatype=datatype, description=description)

    return out


def _extract_knowledge_graph_column_meta(knowledge_graph_data):
    out = {}

    def _ensure_table(table_name):
        t_name = str(table_name or "").strip()
        if not t_name:
            return None
        t_key = t_name.lower()
        if t_key not in out:
            out[t_key] = {}
        return t_key

    def _add_column(table_name, column_name, datatype="", description=""):
        t_key = _ensure_table(table_name)
        c_name = str(column_name or "").strip()
        if t_key is None or not c_name:
            return
        c_key = c_name.lower()
        existing = out[t_key].get(c_key, {})
        out[t_key][c_key] = {
            "datatype": str(datatype or existing.get("datatype") or "").strip(),
            "description": str(description or existing.get("description") or "").strip(),
        }

    def _consume_table_block(table_name, table_payload):
        if not table_name or not isinstance(table_payload, (dict, list)):
            return
        if isinstance(table_payload, list):
            for col_item in table_payload:
                if isinstance(col_item, dict):
                    _add_column(
                        table_name,
                        col_item.get("name") or col_item.get("column") or col_item.get("column_name"),
                        datatype=col_item.get("datatype") or col_item.get("data_type") or col_item.get("type"),
                        description=col_item.get("description") or col_item.get("desc"),
                    )
                else:
                    _add_column(table_name, col_item)
            return

        candidate_columns = (
            table_payload.get("columns")
            or table_payload.get("fields")
            or table_payload.get("column_details")
        )
        if isinstance(candidate_columns, list):
            for col_item in candidate_columns:
                if isinstance(col_item, dict):
                    _add_column(
                        table_name,
                        col_item.get("name") or col_item.get("column") or col_item.get("column_name"),
                        datatype=col_item.get("datatype") or col_item.get("data_type") or col_item.get("type"),
                        description=col_item.get("description") or col_item.get("desc"),
                    )
                else:
                    _add_column(table_name, col_item)
        elif isinstance(candidate_columns, dict):
            for col_name, col_meta in candidate_columns.items():
                if isinstance(col_meta, dict):
                    _add_column(
                        table_name,
                        col_name,
                        datatype=col_meta.get("datatype") or col_meta.get("data_type") or col_meta.get("type"),
                        description=col_meta.get("description") or col_meta.get("desc"),
                    )
                else:
                    _add_column(table_name, col_name, description=col_meta)
        else:
            for col_name, col_meta in table_payload.items():
                if isinstance(col_meta, dict):
                    _add_column(
                        table_name,
                        col_name,
                        datatype=col_meta.get("datatype") or col_meta.get("data_type") or col_meta.get("type"),
                        description=col_meta.get("description") or col_meta.get("desc"),
                    )

    if isinstance(knowledge_graph_data, dict):
        nested_tables = knowledge_graph_data.get("tables")
        if isinstance(nested_tables, list):
            for table_item in nested_tables:
                if not isinstance(table_item, dict):
                    continue
                t_name = table_item.get("table_name") or table_item.get("table") or table_item.get("name")
                _consume_table_block(t_name, table_item)
        elif isinstance(nested_tables, dict):
            for t_name, t_payload in nested_tables.items():
                _consume_table_block(t_name, t_payload)

        for t_name, t_payload in knowledge_graph_data.items():
            if t_name == "tables":
                continue
            if isinstance(t_payload, (dict, list)):
                _consume_table_block(t_name, t_payload)
    elif isinstance(knowledge_graph_data, list):
        for table_item in knowledge_graph_data:
            if not isinstance(table_item, dict):
                continue
            t_name = table_item.get("table_name") or table_item.get("table") or table_item.get("name")
            _consume_table_block(t_name, table_item)

    return out


def _extract_relationship_lines(relationships):
    lines = []

    def _parse_table_col_ref(value):
        text = str(value or "").strip()
        if "." not in text:
            return "", ""
        left, right = text.split(".", 1)
        return left.strip(), right.strip()

    def _append_rel(left_table, left_col, right_table, right_col, relation_type):
        l_table = str(left_table or "").strip()
        l_col = str(left_col or "").strip()
        r_table = str(right_table or "").strip()
        r_col = str(right_col or "").strip()
        if not (l_table and l_col and r_table and r_col):
            return
        rel_type = str(relation_type or "unknown").strip() or "unknown"
        lines.append(f"{l_table}.{l_col} \u2192 {r_table}.{r_col} (type: {rel_type})")

    candidates = []
    if isinstance(relationships, list):
        candidates.extend(relationships)
    elif isinstance(relationships, dict):
        for key in ("relationships", "joins", "edges", "data"):
            nested = relationships.get(key)
            if isinstance(nested, list):
                candidates.extend(nested)
        if not candidates:
            candidates.append(relationships)

    for rel in candidates:
        if not isinstance(rel, dict):
            continue

        left_table = rel.get("left_table") or rel.get("source_table") or rel.get("from_table") or rel.get("table1")
        left_col = rel.get("left_column") or rel.get("source_column") or rel.get("from_column") or rel.get("col1")
        right_table = rel.get("right_table") or rel.get("target_table") or rel.get("to_table") or rel.get("table2")
        right_col = rel.get("right_column") or rel.get("target_column") or rel.get("to_column") or rel.get("col2")
        relation_type = rel.get("relationship_type") or rel.get("cardinality") or rel.get("type") or rel.get("join_type")

        if not (left_table and left_col and right_table and right_col):
            source_ref = rel.get("source") or rel.get("from")
            target_ref = rel.get("target") or rel.get("to")
            if isinstance(source_ref, dict):
                left_table = left_table or source_ref.get("table")
                left_col = left_col or source_ref.get("column")
            elif isinstance(source_ref, str):
                parsed_table, parsed_col = _parse_table_col_ref(source_ref)
                left_table = left_table or parsed_table
                left_col = left_col or parsed_col

            if isinstance(target_ref, dict):
                right_table = right_table or target_ref.get("table")
                right_col = right_col or target_ref.get("column")
            elif isinstance(target_ref, str):
                parsed_table, parsed_col = _parse_table_col_ref(target_ref)
                right_table = right_table or parsed_table
                right_col = right_col or parsed_col

        _append_rel(left_table, left_col, right_table, right_col, relation_type)

    return lines


def _extract_relationship_join_groups(relationships):
    groups = {}

    def _parse_table_col_ref(value):
        text = str(value or "").strip()
        if "." not in text:
            return "", ""
        left, right = text.split(".", 1)
        return left.strip(), right.strip()

    def _append_rel(left_table, left_col, right_table, right_col, relation_type):
        l_table = str(left_table or "").strip()
        l_col = str(left_col or "").strip()
        r_table = str(right_table or "").strip()
        r_col = str(right_col or "").strip()
        if not (l_table and l_col and r_table and r_col):
            return

        rel_type = str(relation_type or "unknown").strip() or "unknown"
        group_key = (l_table.lower(), r_table.lower(), rel_type.lower())
        if group_key not in groups:
            groups[group_key] = {
                "left_table": l_table,
                "right_table": r_table,
                "type": rel_type,
                "keys": [],
                "seen": set(),
            }

        key_sig = f"{l_col.lower()}::{r_col.lower()}"
        if key_sig in groups[group_key]["seen"]:
            return
        groups[group_key]["seen"].add(key_sig)
        groups[group_key]["keys"].append((l_col, r_col))

    candidates = []
    if isinstance(relationships, list):
        candidates.extend(relationships)
    elif isinstance(relationships, dict):
        for key in ("relationships", "joins", "edges", "data"):
            nested = relationships.get(key)
            if isinstance(nested, list):
                candidates.extend(nested)
        if not candidates:
            candidates.append(relationships)

    for rel in candidates:
        if not isinstance(rel, dict):
            continue

        left_table = rel.get("left_table") or rel.get("source_table") or rel.get("from_table") or rel.get("table1")
        left_col = rel.get("left_column") or rel.get("source_column") or rel.get("from_column") or rel.get("col1")
        right_table = rel.get("right_table") or rel.get("target_table") or rel.get("to_table") or rel.get("table2")
        right_col = rel.get("right_column") or rel.get("target_column") or rel.get("to_column") or rel.get("col2")
        relation_type = rel.get("relationship_type") or rel.get("cardinality") or rel.get("type") or rel.get("join_type")

        if not (left_table and left_col and right_table and right_col):
            source_ref = rel.get("source") or rel.get("from")
            target_ref = rel.get("target") or rel.get("to")
            if isinstance(source_ref, dict):
                left_table = left_table or source_ref.get("table")
                left_col = left_col or source_ref.get("column")
            elif isinstance(source_ref, str):
                parsed_table, parsed_col = _parse_table_col_ref(source_ref)
                left_table = left_table or parsed_table
                left_col = left_col or parsed_col

            if isinstance(target_ref, dict):
                right_table = right_table or target_ref.get("table")
                right_col = right_col or target_ref.get("column")
            elif isinstance(target_ref, str):
                parsed_table, parsed_col = _parse_table_col_ref(target_ref)
                right_table = right_table or parsed_table
                right_col = right_col or parsed_col

        _append_rel(left_table, left_col, right_table, right_col, relation_type)

    out = []
    for key in sorted(groups.keys()):
        payload = groups[key]
        keys = payload.get("keys") or []
        join_parts = [
            f"{payload['left_table']}.{l_col} = {payload['right_table']}.{r_col}"
            for l_col, r_col in keys
        ]
        out.append(
            {
                "left_table": payload["left_table"],
                "right_table": payload["right_table"],
                "type": payload["type"],
                "keys": list(keys),
                "join_sql": " AND ".join(join_parts),
                "key_count": len(join_parts),
            }
        )
    return out


def _extract_metric_rows(metrics_data):
    rows = []

    def _append_metric(metric_name, metric_payload):
        name = str(metric_name or "").strip()
        formula = ""
        description = ""
        if isinstance(metric_payload, dict):
            formula = (
                metric_payload.get("formula")
                or metric_payload.get("sql")
                or metric_payload.get("expression")
                or metric_payload.get("metric_sql")
                or ""
            )
            description = metric_payload.get("description") or metric_payload.get("desc") or ""
            if not name:
                name = str(metric_payload.get("metric_name") or metric_payload.get("name") or "").strip()
        elif metric_payload is not None:
            formula = str(metric_payload).strip()

        if not name:
            name = "Unnamed Metric"
        rows.append(
            {
                "name": name,
                "formula": str(formula or "").strip(),
                "description": str(description or "").strip(),
            }
        )

    if isinstance(metrics_data, dict):
        nested_metrics = metrics_data.get("metrics")
        if isinstance(nested_metrics, list):
            for item in nested_metrics:
                if isinstance(item, dict):
                    _append_metric(item.get("metric_name") or item.get("name"), item)
                else:
                    _append_metric("", item)
        else:
            for metric_name, payload in metrics_data.items():
                _append_metric(metric_name, payload)
    elif isinstance(metrics_data, list):
        for item in metrics_data:
            if isinstance(item, dict):
                _append_metric(item.get("metric_name") or item.get("name"), item)
            else:
                _append_metric("", item)

    return rows


def _extract_metrics_from_kb(kb_data):
    if not isinstance(kb_data, dict):
        return {}

    metrics_data = kb_data.get("metrics_data")
    if metrics_data is None:
        return {}

    out = {}

    def _infer_unit(description_text):
        desc_l = str(description_text or "").lower()
        if "rs unit" in desc_l:
            return "currency"
        if "%" in desc_l or "percent" in desc_l:
            return "percent"
        if "no unit" in desc_l or "absolute count" in desc_l:
            return "count"
        return "value"

    def _append_metric(metric_name, metric_payload):
        name = str(metric_name or "").strip()
        formula = ""
        description = ""

        if isinstance(metric_payload, dict):
            if not name:
                name = str(metric_payload.get("metric_name") or metric_payload.get("name") or "").strip()
            raw_formula = (
                metric_payload.get("formula")
                or metric_payload.get("sql")
                or metric_payload.get("expression")
                or metric_payload.get("metric_sql")
                or ""
            )
            formula = "" if raw_formula is None else str(raw_formula)
            raw_description = metric_payload.get("description")
            if raw_description is None:
                raw_description = metric_payload.get("desc")
            description = "" if raw_description is None else str(raw_description).strip()
        elif metric_payload is not None:
            formula = str(metric_payload)

        if not name:
            return

        out[name.lower()] = {
            "name": name,
            "formula": formula,
            "description": description,
            "unit": _infer_unit(description),
        }

    if isinstance(metrics_data, dict):
        nested_metrics = metrics_data.get("metrics")
        if isinstance(nested_metrics, list):
            for item in nested_metrics:
                if isinstance(item, dict):
                    _append_metric(item.get("metric_name") or item.get("name"), item)
                else:
                    _append_metric("", item)
        else:
            for metric_name, payload in metrics_data.items():
                _append_metric(metric_name, payload)
    elif isinstance(metrics_data, list):
        for item in metrics_data:
            if isinstance(item, dict):
                _append_metric(item.get("metric_name") or item.get("name"), item)
            else:
                _append_metric("", item)

    return out


def _detect_requested_metric(user_prompt, metrics_dict):
    prompt_l = str(user_prompt or "").strip().lower()
    metrics = metrics_dict if isinstance(metrics_dict, dict) else {}
    if not prompt_l or not metrics:
        return None

    alias_groups = [
        ("fill rate", ["fill rate", "fillrate", "fill_rate"]),
        ("suspicious eco", ["suspicious eco", "suspicious", "suspicous", "suspecious", "rag"]),
        ("eco", ["eco", "effective outlet", "outlet coverage"]),
        ("ulpo", ["ulpo", "unique lines", "lines per outlet"]),
        ("abv", ["abv", "average bill", "bill value"]),
        ("drr", ["drr", "daily run rate", "daily rate"]),
        ("sales", ["sales", "gross value", "revenue"]),
        ("units", ["units", "quantity", "volume"]),
    ]

    # Prefer explicit metric-name matches first.
    for metric_key, metric_payload in metrics.items():
        metric_name = str((metric_payload or {}).get("name") or metric_key or "").strip().lower()
        if metric_name == "eco" and any(tok in prompt_l for tok in ["suspicious", "suspicous", "suspecious", "rag"]):
            continue
        if metric_name and metric_name in prompt_l:
            return metric_payload if isinstance(metric_payload, dict) else None

    # Then fallback to alias dictionary.
    for canonical_key, aliases in alias_groups:
        if any(alias in prompt_l for alias in aliases):
            hit = metrics.get(canonical_key)
            if isinstance(hit, dict):
                return hit
            # Fallback: compare against payload "name" if key differs.
            for metric_payload in metrics.values():
                if not isinstance(metric_payload, dict):
                    continue
                metric_name = str(metric_payload.get("name") or "").strip().lower()
                if metric_name == canonical_key:
                    return metric_payload
            return None

    return None


def _validate_metric_formula_in_sql(sql, detected_metric):
    if not isinstance(detected_metric, dict):
        return True

    metric_name = str(detected_metric.get("name") or "").strip()
    metric_name_l = metric_name.lower()
    sql_l = str(sql or "").lower()

    is_valid = True

    if metric_name_l == "fill rate":
        is_valid = ("sum(adjusted_inv_quantity)" in sql_l) and ("sum(adjusted_ord_quantity)" in sql_l)
    elif metric_name_l == "eco":
        is_valid = "count(distinct retaileruid)" in sql_l
    elif metric_name_l == "ulpo":
        has_distinct_material = "count(distinct material_no)" in sql_l
        has_retailer = "retaileruid" in sql_l
        has_ulpo_agg = "avg(unique_lines_per_retailer)" in sql_l or "avg(" in sql_l
        is_valid = has_distinct_material and has_retailer and has_ulpo_agg
    elif metric_name_l == "abv":
        has_concat_invoice = "concat(aw_code,'_',invoicenumber)" in sql_l or "concat(aw_code, '_', invoicenumber)" in sql_l
        has_distinct_invoice_count = (
            "count(distinct invoicenumber)" in sql_l
            or "count(distinct concat(aw_code,'_',invoicenumber))" in sql_l
            or "count(distinct concat(aw_code, '_', invoicenumber))" in sql_l
        )
        is_valid = has_concat_invoice or has_distinct_invoice_count
    elif metric_name_l == "suspicious eco":
        has_rag_filter = "ret_rag_flag = 'r'" in sql_l or "ret_rag_flag='r'" in sql_l
        has_distinct_retailer = (
            "count(distinct inv_retailer)" in sql_l
            or "count(distinct suspicious_inv_retailer)" in sql_l
            or "count(distinct retaileruid)" in sql_l
        )
        is_valid = has_rag_filter and has_distinct_retailer
    elif metric_name_l == "drr":
        has_sales_component = "sum(gross_value)" in sql_l or "sum(" in sql_l
        has_day_count_component = (
            "count(distinct billdate)" in sql_l
            or "count(*) over" in sql_l
            or "count(1) over" in sql_l
            or "count(*) filter" in sql_l
        )

        grouped_by_billdate = bool(re.search(r"(?is)\bgroup\s+by\b.*\bbilldate\b", sql_l))
        count_distinct_billdate = "count(distinct billdate)" in sql_l
        has_window = "over (" in sql_l

        # Invalid DRR pattern: grouping by BillDate while dividing by COUNT(DISTINCT BillDate)
        # without a window/cumulative frame. This collapses denominator to 1 at each point.
        invalid_collapsed_denominator = grouped_by_billdate and count_distinct_billdate and not has_window
        is_valid = has_sales_component and has_day_count_component and (not invalid_collapsed_denominator)

    if not is_valid:
        print(
            f"[METRIC GUARD] Generated SQL missing key formula components for metric {metric_name}. Rebuilding SQL from formula template.",
            flush=True,
        )
    return is_valid


def _build_sql_from_metric_formula(metric, dimension_col, dimension_table=None, limit=20, order='DESC'):
    metric_payload = metric if isinstance(metric, dict) else {}
    formula_raw = metric_payload.get("formula")
    formula = "" if formula_raw is None else str(formula_raw).strip().rstrip(";")
    dim_col = str(dimension_col or "").strip()
    dim_table = str(dimension_table or "").strip()
    if not formula or not dim_col:
        return ""

    order_dir = str(order or "DESC").strip().upper()
    if order_dir not in {"ASC", "DESC"}:
        order_dir = "DESC"
    try:
        limit_n = int(limit)
    except Exception:
        limit_n = 20
    if limit_n <= 0:
        limit_n = 20

    dim_name_l = str(dim_col or "").strip().strip("`").lower()
    time_tokens = {"date", "day", "week", "month", "year", "time", "timestamp"}
    is_time_dimension = any(tok in dim_name_l for tok in time_tokens)
    order_clause = "ORDER BY 1 ASC" if is_time_dimension else f"ORDER BY 2 {order_dir}"
    limit_clause = "" if is_time_dimension else f"\nLIMIT {limit_n}"

    metric_name_l = str(metric_payload.get("name") or "").strip().lower()

    def _strip_table_prefix(col_expr):
        raw = str(col_expr or "").strip().strip("`")
        if "." in raw:
            raw = raw.split(".")[-1]
        raw = raw.strip("`")
        return raw

    # ULPO is "avg(unique lines per retailer per time period)".
    # For time charts, force grain-safe SQL:
    # 1) compute COUNT(DISTINCT item) by (time, retailer)
    # 2) AVG that per time bucket
    if metric_name_l == "ulpo" and is_time_dimension:
        item_col = ""
        retailer_col = ""

        item_match = re.search(r"(?is)count\s*\(\s*distinct\s+([A-Za-z0-9_`\.]+)\s*\)", formula)
        if item_match:
            item_col = _strip_table_prefix(item_match.group(1))

        grp_match = re.search(r"(?is)group\s+by\s+([A-Za-z0-9_`\.]+)", formula)
        if grp_match:
            retailer_col = _strip_table_prefix(grp_match.group(1))

        if not item_col:
            item_col = "Material_No"
        if not retailer_col:
            retailer_col = "RetailerUID"

        dim_safe = _strip_table_prefix(dim_col) or dim_col
        return (
            f"SELECT CAST({dim_safe} AS DATE) AS x, AVG(unique_lines_per_retailer) AS y\n"
            f"FROM (\n"
            f"  SELECT CAST({dim_safe} AS DATE) AS {dim_safe}, {retailer_col}, COUNT(DISTINCT {item_col}) AS unique_lines_per_retailer\n"
            f"  FROM analysis_view\n"
            f"  WHERE {dim_safe} IS NOT NULL AND {retailer_col} IS NOT NULL AND {item_col} IS NOT NULL\n"
            f"  GROUP BY 1, 2\n"
            f") ulpo_grain\n"
            f"GROUP BY 1\n"
            f"ORDER BY 1 ASC"
        )

    # DRR is "sales / distinct active days". For time charts at day grain, avoid
    # denominator collapse (COUNT DISTINCT day == 1 in each grouped row) by producing
    # cumulative run-rate across the selected period.
    if metric_name_l == "drr" and is_time_dimension:
        dim_safe = _strip_table_prefix(dim_col) or dim_col
        return (
            f"WITH drr_daily AS (\n"
            f"  SELECT CAST({dim_safe} AS DATE) AS x, SUM(Gross_Value) AS daily_sales\n"
            f"  FROM analysis_view\n"
            f"  WHERE {dim_safe} IS NOT NULL\n"
            f"  GROUP BY 1\n"
            f")\n"
            f"SELECT x,\n"
            f"       SUM(daily_sales) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)\n"
            f"       / NULLIF(COUNT(*) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) AS y\n"
            f"FROM drr_daily\n"
            f"ORDER BY 1 ASC"
        )

    # Suspicious ECO is typically evaluated at a period grain. For time trends,
    # force month-level ratio to avoid sparse day-level zero-heavy lines.
    if metric_name_l == "suspicious eco" and is_time_dimension:
        dim_safe = _strip_table_prefix(dim_col) or dim_col
        return (
            f"SELECT DATE_TRUNC('month', CAST({dim_safe} AS DATE)) AS x,\n"
            f"       COUNT(DISTINCT CASE WHEN suspicious_ret_rag_flag = 'R' THEN RetailerUID END)\n"
            f"       / NULLIF(COUNT(DISTINCT RetailerUID), 0) AS y\n"
            f"FROM analysis_view\n"
            f"WHERE {dim_safe} IS NOT NULL AND RetailerUID IS NOT NULL\n"
            f"GROUP BY 1\n"
            f"ORDER BY 1 ASC"
        )

    def _extract_relationship_join_sql_from_cache(left_table, right_table):
        left_l = str(left_table or "").strip().lower()
        right_l = str(right_table or "").strip().lower()
        if not left_l or not right_l:
            return ""
        if not isinstance(_kb_cache, dict):
            return ""

        for cache_entry in _kb_cache.values():
            kb_obj = cache_entry[0] if isinstance(cache_entry, tuple) and cache_entry else cache_entry
            if not isinstance(kb_obj, dict):
                continue
            rel_groups = _extract_relationship_join_groups(kb_obj.get("relationships"))
            for rel in rel_groups:
                if not isinstance(rel, dict):
                    continue
                l_tbl = str(rel.get("left_table") or "").strip().lower()
                r_tbl = str(rel.get("right_table") or "").strip().lower()
                join_sql = str(rel.get("join_sql") or "").strip()
                if not join_sql:
                    continue
                if l_tbl == left_l and r_tbl == right_l:
                    return join_sql
                if l_tbl == right_l and r_tbl == left_l:
                    # reverse join direction
                    keys = rel.get("keys") or []
                    parts = [f"{right_table}.{r_col} = {left_table}.{l_col}" for l_col, r_col in keys if l_col and r_col]
                    return " AND ".join(parts) if parts else join_sql
        return ""

    def _append_dimension_join_if_needed(from_clause):
        clause = str(from_clause or "").strip()
        if not clause or not dim_table:
            return clause
        dim_l = dim_table.lower()
        if re.search(rf"(?i)\b{re.escape(dim_l)}\b", clause.lower()):
            return clause

        from_match = re.search(r"(?is)\bfrom\s+([A-Za-z0-9_`\.]+)", clause)
        base_table = str(from_match.group(1) or "").strip("`").split(".")[-1] if from_match else ""
        join_sql = _extract_relationship_join_sql_from_cache(base_table, dim_table)
        if not join_sql:
            return clause
        return f"{clause}\nLEFT JOIN {dim_table} ON {join_sql}"

    def _append_dimension_not_null(clause_sql):
        clause = str(clause_sql or "").strip()
        if not clause:
            return clause
        if re.search(r"(?is)\bwhere\b", clause):
            return f"{clause}\n  AND {dim_col} IS NOT NULL"
        return f"{clause}\nWHERE {dim_col} IS NOT NULL"

    if formula.lower().startswith(("select ", "with ")):
        metric_expr = _extract_primary_metric_expression_from_sql(formula) or str(metric_payload.get("formula") or "").strip()
        from_match = re.search(r"(?is)\bfrom\b", formula)
        if not from_match:
            return ""
        from_tail = formula[from_match.start():].strip()
        truncate_match = re.search(r"(?is)\bgroup\s+by\b|\border\s+by\b|\blimit\b", from_tail)
        if truncate_match:
            from_tail = from_tail[:truncate_match.start()].strip()
        from_tail = _append_dimension_join_if_needed(from_tail)
        from_tail = _append_dimension_not_null(from_tail)
        return (
            f"SELECT {dim_col} AS x, {metric_expr} AS y\n"
            f"{from_tail}\n"
            f"GROUP BY 1\n"
            f"{order_clause}"
            f"{limit_clause}"
        )

    from_clause = f"FROM analysis_view"
    from_clause = _append_dimension_join_if_needed(from_clause)
    from_clause = _append_dimension_not_null(from_clause)
    return (
        f"SELECT {dim_col} AS x, {formula} AS y\n"
        f"{from_clause}\n"
        f"GROUP BY 1\n"
        f"{order_clause}"
        f"{limit_clause}"
    )


def _extract_relationships_from_kb(kb_data):
    if not isinstance(kb_data, dict):
        return {}

    rel_groups = _extract_relationship_join_groups(kb_data.get("relationships"))
    if not isinstance(rel_groups, list) or not rel_groups:
        return {}

    selected_columns_map = _normalize_selected_columns_map(kb_data.get("selected_columns"))
    out = {}
    for rel in rel_groups:
        if not isinstance(rel, dict):
            continue
        left_table = str(rel.get("left_table") or "").strip()
        right_table = str(rel.get("right_table") or "").strip()
        if not left_table or not right_table:
            continue
        key = (left_table.lower(), right_table.lower())

        rel_type = str(rel.get("type") or "unknown").strip() or "unknown"
        raw_keys = rel.get("keys") or rel.get("join_keys") or []
        join_keys = []
        for pair in raw_keys:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                left_col = str(pair[0] or "").strip()
                right_col = str(pair[1] or "").strip()
                if left_col and right_col:
                    join_keys.append((left_col, right_col))
            elif isinstance(pair, dict):
                left_col = str(pair.get("left_column") or pair.get("left_col") or "").strip()
                right_col = str(pair.get("right_column") or pair.get("right_col") or "").strip()
                if left_col and right_col:
                    join_keys.append((left_col, right_col))

        if not join_keys:
            join_sql_text = str(rel.get("join_sql") or "").strip()
            for part in [p.strip() for p in re.split(r"(?i)\band\b", join_sql_text) if p.strip()]:
                m = re.match(rf"(?i)^\s*{re.escape(left_table)}\.([A-Za-z0-9_`]+)\s*=\s*{re.escape(right_table)}\.([A-Za-z0-9_`]+)\s*$", part)
                if m:
                    join_keys.append((str(m.group(1)).strip("`"), str(m.group(2)).strip("`")))

        if not join_keys:
            continue

        join_keys = _optimize_join_key_pairing(join_keys)
        left_table_key = left_table.lower()
        right_table_key = right_table.lower()
        left_candidates = [
            str((c or {}).get("name") or "").strip()
            for c in ((selected_columns_map.get(left_table_key) or {}).get("columns") or [])
            if str((c or {}).get("name") or "").strip()
        ] if isinstance(selected_columns_map, dict) else []
        right_candidates = [
            str((c or {}).get("name") or "").strip()
            for c in ((selected_columns_map.get(right_table_key) or {}).get("columns") or [])
            if str((c or {}).get("name") or "").strip()
        ] if isinstance(selected_columns_map, dict) else []
        join_keys = _repair_join_key_semantics(
            join_keys,
            available_left_columns=left_candidates,
            available_right_columns=right_candidates,
        )

        existing = out.get(key)
        if existing and isinstance(existing, dict):
            seen = {f"{a.lower()}::{b.lower()}" for a, b in existing.get("join_keys", []) if a and b}
            for a, b in join_keys:
                sig = f"{a.lower()}::{b.lower()}"
                if sig not in seen:
                    existing["join_keys"].append((a, b))
                    seen.add(sig)
            existing["join_sql"] = " AND ".join(
                f"{left_table}.{a} = {right_table}.{b}" for a, b in existing.get("join_keys", [])
            )
            continue

        out[key] = {
            "type": rel_type,
            "join_keys": list(join_keys),
            "join_sql": " AND ".join(f"{left_table}.{a} = {right_table}.{b}" for a, b in join_keys),
        }

    return out


def _validate_joins_in_sql(sql, relationships_dict):
    sql_text = str(sql or "")
    if not sql_text:
        return sql_text
    if not isinstance(relationships_dict, dict) or not relationships_dict:
        return sql_text

    fixed_sql = sql_text

    def _inject_condition_into_join_clause(query_sql, join_table, condition):
        pattern = re.compile(
            rf"(?is)(\b(?:left|inner|right|full|cross)?\s*join\s+{re.escape(join_table)}\b\s+on\s+)(.*?)(?=(\bleft\b|\binner\b|\bright\b|\bfull\b|\bcross\b)?\s*join\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$)"
        )
        m = pattern.search(query_sql)
        if not m:
            return query_sql, False
        prefix = m.group(1)
        join_on_expr = str(m.group(2) or "").strip()
        updated_on = f"{join_on_expr} AND {condition}" if join_on_expr else condition
        replaced = query_sql[:m.start()] + f"{prefix}{updated_on}" + query_sql[m.end():]
        return replaced, True

    def _remove_condition_from_join_clause(query_sql, join_table, condition):
        pattern = re.compile(
            rf"(?is)(\b(?:left|inner|right|full|cross)?\s*join\s+{re.escape(join_table)}\b\s+on\s+)(.*?)(?=(\bleft\b|\binner\b|\bright\b|\bfull\b|\bcross\b)?\s*join\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$)"
        )
        m = pattern.search(query_sql)
        if not m:
            return query_sql, False
        prefix = m.group(1)
        join_on_expr = str(m.group(2) or "").strip()
        if not join_on_expr:
            return query_sql, False

        cond_parts = [p.strip() for p in re.split(r"(?i)\band\b", join_on_expr) if p.strip()]
        kept = []
        removed = False
        cond_norm = re.sub(r"\s+", "", str(condition or "").lower())
        for part in cond_parts:
            part_norm = re.sub(r"\s+", "", part.lower())
            if part_norm == cond_norm:
                removed = True
                continue
            kept.append(part)

        if not removed:
            return query_sql, False

        rebuilt = " AND ".join(kept) if kept else "1=1"
        replaced = query_sql[:m.start()] + f"{prefix}{rebuilt}" + query_sql[m.end():]
        return replaced, True

    for rel_key, rel_payload in relationships_dict.items():
        if not isinstance(rel_key, (list, tuple)) or len(rel_key) < 2:
            continue
        if not isinstance(rel_payload, dict):
            continue
        left_table = str(rel_key[0] or "").strip()
        right_table = str(rel_key[1] or "").strip()
        if not left_table or not right_table:
            continue

        if not re.search(rf"(?i)\b{re.escape(right_table)}\b", fixed_sql):
            continue

        join_keys = rel_payload.get("join_keys") or []
        required_conditions = []
        for pair in join_keys:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                l_col = str(pair[0] or "").strip()
                r_col = str(pair[1] or "").strip()
                if l_col and r_col:
                    required_conditions.append(f"{left_table}.{l_col} = {right_table}.{r_col}")

        if not required_conditions:
            join_sql_text = str(rel_payload.get("join_sql") or "").strip()
            if join_sql_text:
                required_conditions = [part.strip() for part in re.split(r"(?i)\band\b", join_sql_text) if part.strip()]

        # Remove contradictory cross-mapped predicates first. Example:
        # expected: A.billdate = B.yearmonth and A.retaileruid = B.inv_retailer
        # wrong:    A.billdate = B.inv_retailer (kept previously and forced 0 rows)
        pair_map = {}
        left_cols = set()
        right_cols = set()
        for pair in join_keys:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                l_col = str(pair[0] or "").strip()
                r_col = str(pair[1] or "").strip()
                if l_col and r_col:
                    pair_map[l_col.lower()] = r_col.lower()
                    left_cols.add(l_col.lower())
                    right_cols.add(r_col.lower())

        if pair_map:
            join_match = re.search(
                rf"(?is)\b(?:left|inner|right|full|cross)?\s*join\s+{re.escape(right_table)}\b\s+on\s+(.*?)(?=(\bleft\b|\binner\b|\bright\b|\bfull\b|\bcross\b)?\s*join\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$)",
                fixed_sql,
            )
            join_on_expr = str(join_match.group(1) or "") if join_match else ""
            join_parts = [p.strip() for p in re.split(r"(?i)\band\b", join_on_expr) if p.strip()]
            for part in join_parts:
                m = re.match(
                    rf"(?is)^\s*{re.escape(left_table)}\.([A-Za-z0-9_`]+)\s*=\s*{re.escape(right_table)}\.([A-Za-z0-9_`]+)\s*$",
                    part,
                )
                if not m:
                    continue
                l_found = str(m.group(1) or "").strip("`").lower()
                r_found = str(m.group(2) or "").strip("`").lower()
                expected_r = pair_map.get(l_found)
                if l_found in left_cols and r_found in right_cols and expected_r and r_found != expected_r:
                    print(
                        f"[JOIN GUARD] Table {right_table} has conflicting join key mapping {part}. Replacing with KB mapping.",
                        flush=True,
                    )
                    fixed_sql, removed = _remove_condition_from_join_clause(fixed_sql, right_table, part)
                    if removed:
                        print(f"[JOIN FIX] Added missing join key: {left_table}.{l_found} = {right_table}.{expected_r}", flush=True)

        for condition in required_conditions:
            if re.search(re.escape(condition), fixed_sql, flags=re.IGNORECASE):
                continue
            print(
                f"[JOIN GUARD] Table {right_table} found in SQL but join key {condition} is missing. Injecting missing key.",
                flush=True,
            )
            fixed_sql, injected = _inject_condition_into_join_clause(fixed_sql, right_table, condition)
            if injected:
                print(f"[JOIN FIX] Added missing join key: {condition}", flush=True)

    return fixed_sql


_METRIC_FORMULA_IGNORE_TOKENS = {
    "select", "from", "where", "group", "by", "order", "limit", "having", "as", "and", "or",
    "not", "on", "case", "when", "then", "else", "end", "distinct", "over", "partition",
    "sum", "count", "avg", "min", "max", "coalesce", "cast", "round", "nullif", "ifnull",
    "greatest", "least", "date", "year", "month", "day", "week", "quarter",
    "analysis_view", "final_view", "master_view",
}


def _infer_metric_aggregation(formula_text):
    text = str(formula_text or "").strip().lower()
    if "/" in text:
        return "AUTO"
    if "sum(" in text:
        return "SUM"
    if "count(" in text:
        return "COUNT"
    if "avg(" in text:
        return "AVG"
    if "min(" in text:
        return "MIN"
    if "max(" in text:
        return "MAX"
    return "AUTO"


def _metric_formula_looks_like_expression(formula_text):
    text = str(formula_text or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith("select ") or low.startswith("with "):
        return False
    if " from " in low or ";" in text:
        return False
    return True


def _extract_metric_formula_columns(formula_text, available_columns):
    cols = [str(c).strip() for c in (available_columns or []) if str(c).strip()]
    if not cols:
        return []

    col_lookup = {c.lower(): c for c in cols}
    tail_lookup = {}
    for c in cols:
        tail = str(c).lower().split("_")[-1]
        tail_lookup.setdefault(tail, c)

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula_text or ""))
    ordered = []
    seen = set()
    for tok in tokens:
        key = str(tok or "").strip().lower()
        if not key or key in _METRIC_FORMULA_IGNORE_TOKENS:
            continue
        resolved = col_lookup.get(key)
        if not resolved:
            resolved = tail_lookup.get(key)
        if not resolved:
            continue
        rkey = resolved.lower()
        if rkey in seen:
            continue
        seen.add(rkey)
        ordered.append(resolved)
    return ordered


def _extract_primary_metric_expression_from_sql(formula_text):
    sql = str(formula_text or "").strip()
    if not sql:
        return ""
    sql = re.sub(r";+\s*$", "", sql)

    # Capture the final SELECT list before the corresponding FROM.
    select_blocks = re.findall(r"(?is)\bselect\b\s+(.*?)\s+\bfrom\b", sql)
    if not select_blocks:
        return ""
    select_list = str(select_blocks[-1] or "").strip()
    if not select_list:
        return ""

    # Split by top-level commas and pick the first expression.
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(select_list):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(select_list[start:i].strip())
            start = i + 1
    parts.append(select_list[start:].strip())
    expr = str(parts[0] if parts else select_list).strip()

    # Remove trailing alias.
    expr = re.sub(r"(?is)\s+as\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", expr).strip()
    expr = re.sub(r"(?is)\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", expr).strip() if " " in expr and ")" in expr else expr
    return expr


def _normalize_metric_expression_for_analysis_view(expr_text, available_columns):
    expr = str(expr_text or "").strip()
    if not expr:
        return ""
    cols = [str(c).strip() for c in (available_columns or []) if str(c).strip()]
    if not cols:
        return expr

    probe_sql = f"SELECT {expr} AS value FROM {DATABRICKS_LOGICAL_VIEW_NAME}"
    normalized_sql, _ = _enforce_analysis_view_sql_contract(
        probe_sql,
        available_columns=cols,
    )
    m = re.search(
        rf"(?is)\bselect\b\s+(.*?)\s+\bas\b\s+value\s+\bfrom\b\s+{re.escape(DATABRICKS_LOGICAL_VIEW_NAME)}\b",
        str(normalized_sql or ""),
    )
    return str(m.group(1)).strip() if m else expr


def _infer_metric_unit_type(metric_name="", description="", formula_text="", expression_text=""):
    name = str(metric_name or "").strip().lower()
    desc = str(description or "").strip().lower()
    formula = str(formula_text or "").strip().lower()
    expr = str(expression_text or "").strip().lower()
    blob = " ".join([name, desc, formula, expr])

    # Strong metric-name overrides first.
    if any(tok in name for tok in ["suspicious eco", "fill rate"]):
        return "percent"
    if "drr" in name or "daily run rate" in name:
        return "currency_rate"
    if any(tok in name for tok in ["abv", "average bill", "sales"]):
        return "currency"
    if any(tok in name for tok in ["eco", "ulpo", "units", "distinct retailer"]):
        return "count"

    if any(tok in blob for tok in ["%", "percent", "percentage", "fill rate", "suspicious eco"]):
        return "percent"
    if any(tok in blob for tok in ["rs", "rupee", "inr", "currency", "sales", "abv"]):
        return "currency"
    if "drr" in blob or "daily run rate" in blob:
        return "currency_rate"
    if any(tok in blob for tok in ["no unit", "absolute count", "count(", "distinct"]):
        return "count"
    if "units" in name:
        return "count"
    return "number"


def _canonical_metric_unit_from_name(metric_name_or_label=""):
    text = str(metric_name_or_label or "").strip().lower()
    if not text:
        return ""
    if "suspicious eco" in text:
        return "percent"
    if "fill rate" in text:
        return "percent"
    if "drr" in text or "daily run rate" in text:
        return "currency_rate"
    if "abv" in text or "average bill value" in text or "average bill" in text:
        return "currency"
    if "eco" in text or "distinct retailer" in text:
        return "count"
    if "ulpo" in text:
        return "count"
    if "units" in text:
        return "count"
    if "sales" in text:
        return "currency"
    return ""


def _resolve_effective_metric_unit(
    metric_name="",
    label_text="",
    description="",
    formula_text="",
    expression_text="",
    explicit_unit="",
    profile_unit="",
):
    valid_units = {"currency", "currency_rate", "count", "percent", "number"}

    canonical = _canonical_metric_unit_from_name(metric_name) or _canonical_metric_unit_from_name(label_text)
    if canonical:
        return canonical

    profile = str(profile_unit or "").strip().lower()
    if profile in valid_units:
        return profile

    explicit = str(explicit_unit or "").strip().lower()
    if explicit in valid_units:
        return explicit

    return _infer_metric_unit_type(
        metric_name=metric_name or label_text,
        description=description,
        formula_text=formula_text,
        expression_text=expression_text,
    )


def _build_kb_metric_profiles(kb_data, schema_columns):
    metric_rows = _extract_metric_rows((kb_data or {}).get("metrics_data")) if isinstance(kb_data, dict) else []
    if not metric_rows:
        return []

    cols = [str(c).strip() for c, _ in (schema_columns or []) if str(c).strip()]
    dtype_lookup = {str(c).lower(): str(t) for c, t in (schema_columns or [])}
    out = []
    seen = set()
    for row in metric_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        formula = str(row.get("formula") or "").strip()
        description = str(row.get("description") or "").strip()
        agg = _infer_metric_aggregation(formula)
        matched_cols = _extract_metric_formula_columns(formula, cols)

        primary_col = ""
        for c in matched_cols:
            if _is_numeric_dtype(dtype_lookup.get(str(c).lower(), "")):
                primary_col = c
                break
        if not primary_col and matched_cols:
            primary_col = matched_cols[0]

        expr = ""
        if _metric_formula_looks_like_expression(formula):
            expr = formula
        elif str(formula).strip().lower().startswith(("select ", "with ")):
            expr = _extract_primary_metric_expression_from_sql(formula)

        if expr:
            expr = _normalize_metric_expression_for_analysis_view(expr, cols)
            extra_cols = _extract_metric_formula_columns(expr, cols)
            for c in extra_cols:
                if str(c).lower() not in {str(x).lower() for x in matched_cols}:
                    matched_cols.append(c)
            if not primary_col:
                for c in extra_cols:
                    if _is_numeric_dtype(dtype_lookup.get(str(c).lower(), "")):
                        primary_col = c
                        break
                if not primary_col and extra_cols:
                    primary_col = extra_cols[0]

        if not expr:
            if primary_col:
                agg_safe = agg if agg in {"SUM", "COUNT", "AVG", "MIN", "MAX"} else ("SUM" if _is_numeric_dtype(dtype_lookup.get(primary_col.lower(), "")) else "COUNT")
                if agg_safe == "COUNT":
                    expr = f"COUNT({_quote_identifier(primary_col)})"
                else:
                    expr = f"{agg_safe}({_quote_identifier(primary_col)})"
            else:
                expr = "COUNT(*)"

        out.append(
            {
                "name": name,
                "formula": formula,
                "description": description,
                "aggregation": agg if agg in {"SUM", "COUNT", "AVG", "MIN", "MAX"} else "AUTO",
                "columns": matched_cols,
                "primary_column": primary_col,
                "expression": expr,
                "unit_type": _infer_metric_unit_type(
                    metric_name=name,
                    description=description,
                    formula_text=formula,
                    expression_text=expr,
                ),
            }
        )
    return out


def _resolve_metric_profile_for_hints(metric_profiles, metric_name="", metric_formula="", preferred_columns=None):
    profiles = metric_profiles if isinstance(metric_profiles, list) else []
    if not profiles:
        return None

    name_hint = str(metric_name or "").strip().lower()
    if name_hint:
        hit = next((m for m in profiles if str((m or {}).get("name") or "").strip().lower() == name_hint), None)
        if hit:
            return hit

    formula_hint = str(metric_formula or "").strip()
    if formula_hint:
        fh_low = formula_hint.lower()
        hit = next((m for m in profiles if str((m or {}).get("formula") or "").strip().lower() == fh_low), None)
        if hit:
            return hit

    pref_cols = [str(c).strip().lower() for c in (preferred_columns or []) if str(c).strip()]
    if pref_cols:
        for pref in pref_cols:
            for profile in profiles:
                cols = [str(c).strip().lower() for c in ((profile or {}).get("columns") or []) if str(c).strip()]
                primary = str((profile or {}).get("primary_column") or "").strip().lower()
                if pref and (pref in cols or (primary and pref == primary)):
                    return profile

    return profiles[0]


def _resolve_default_map_metric_profile(metric_profiles, metric_col=""):
    profiles = metric_profiles if isinstance(metric_profiles, list) else []
    if not profiles:
        return None

    exact_sales = next(
        (
            p for p in profiles
            if str((p or {}).get("name") or "").strip().lower() == "sales"
        ),
        None,
    )
    if exact_sales:
        return exact_sales

    metric_col_l = str(metric_col or "").strip().lower()
    quantity_like = any(tok in metric_col_l for tok in ["qty", "quantity", "unit", "units", "volume"])
    amount_like = any(tok in metric_col_l for tok in ["amount", "value", "sales", "revenue", "gross", "net"])

    if quantity_like:
        preferred_name_tokens = ["units", "unit sold", "volume"]
    elif amount_like:
        preferred_name_tokens = ["sales", "sale", "revenue", "value"]
    else:
        preferred_name_tokens = ["sales", "sale", "revenue", "units", "value"]

    for token in preferred_name_tokens:
        hit = next(
            (
                p for p in profiles
                if token in str((p or {}).get("name") or "").strip().lower()
            ),
            None,
        )
        if hit:
            return hit

    def _score(profile):
        p = profile if isinstance(profile, dict) else {}
        name_l = str(p.get("name") or "").strip().lower()
        formula_l = str(p.get("formula") or "").strip().lower()
        agg_l = str(p.get("aggregation") or "").strip().upper()
        unit_l = str(p.get("unit_type") or "").strip().lower()
        score = 0

        if quantity_like and any(tok in name_l for tok in ["unit", "units", "volume"]):
            score += 200
        if amount_like and any(tok in name_l for tok in ["sales", "sale", "revenue", "value"]):
            score += 200

        if agg_l == "SUM":
            score += 60
        if unit_l in {"currency", "count", "number"}:
            score += 25

        if "/" in formula_l:
            score -= 90
        if any(tok in name_l for tok in ["abv", "drr", "fill rate", "suspicious", "rate", "%"]):
            score -= 120
        if "avg(" in formula_l or "average" in name_l:
            score -= 80

        return score

    ranked = sorted(profiles, key=_score, reverse=True)
    return ranked[0] if ranked else None


def _select_map_metric_profile_with_llm(metric_profiles, region_col="", schema_context="", kb_data=None, logs=None):
    profiles = metric_profiles if isinstance(metric_profiles, list) else []
    if not profiles:
        return None

    metric_candidates = []
    for p in profiles[:40]:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        metric_candidates.append(
            {
                "name": name,
                "formula": str(p.get("formula") or "").strip(),
                "description": str(p.get("description") or "").strip(),
                "aggregation": str(p.get("aggregation") or "").strip().upper(),
                "unit_type": str(p.get("unit_type") or "").strip().lower(),
                "primary_column": str(p.get("primary_column") or "").strip(),
            }
        )
    if not metric_candidates:
        return None

    relationships = _extract_relationship_lines((kb_data or {}).get("relationships")) if isinstance(kb_data, dict) else []
    rel_text = "\n".join(relationships[:30]) if relationships else "Not available."

    system_prompt = (
        "You are selecting ONE KPI metric for an India region map in a BI dashboard.\n"
        "Pick exactly one metric from the provided list.\n"
        "Prefer additive regional metrics (e.g., Sales/Units) over ratio/rate metrics unless clearly better.\n"
        "Dimension is fixed to region names.\n"
        "Return ONLY JSON with keys: metric_name, reason."
    )
    user_prompt = (
        f"Map dimension column (region): {region_col or '(unknown)'}\n\n"
        f"Available metrics (KB Section 4):\n{json.dumps(metric_candidates, ensure_ascii=False)}\n\n"
        f"Schema context:\n{str(schema_context or '')[:12000]}\n\n"
        f"KB relationships:\n{rel_text}\n\n"
        "Select one best metric for a region-wise India map."
    )

    llm_raw, _ = call_ai_with_retry(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        json_mode=True,
        debug_logs=logs,
        context="Select Map Metric",
    )
    try:
        parsed = json.loads(llm_raw) if llm_raw else {}
    except Exception:
        parsed = {}

    selected_name = str((parsed or {}).get("metric_name") or "").strip().lower()
    if not selected_name:
        return None

    for p in profiles:
        if str((p or {}).get("name") or "").strip().lower() == selected_name:
            if logs is not None:
                logs.append(f"[MAP] LLM selected map metric: {str((p or {}).get('name') or '').strip()}")
            return p

    if logs is not None:
        logs.append(f"[WARN] Map metric selected by LLM not found in KB metrics: {selected_name}")
    return None


def _metric_dimension_sql(metric_profile, dimension_col, chart_type, date_column=None, logical_table_name="analysis_view"):
    profile = metric_profile if isinstance(metric_profile, dict) else {}
    dim_col = str(dimension_col or "").strip()
    expr = str(profile.get("expression") or "").strip() or "COUNT(*)"
    ctype = str(chart_type or "bar").strip().lower()

    if dim_col and date_column and str(dim_col).strip().lower() == str(date_column).strip().lower():
        dim_ident = _quote_identifier(dim_col)
        return (
            f"SELECT CAST({dim_ident} AS DATE) AS x, {expr} AS y "
            f"FROM {logical_table_name} "
            f"WHERE {dim_ident} IS NOT NULL "
            "GROUP BY 1 ORDER BY 1"
        )

    if dim_col:
        dim_ident = _quote_identifier(dim_col)
        if ctype == "line":
            return (
                f"SELECT CAST({dim_ident} AS STRING) AS x, {expr} AS y "
                f"FROM {logical_table_name} "
                f"WHERE {dim_ident} IS NOT NULL "
                "GROUP BY 1 ORDER BY 1 LIMIT 30"
            )
        return (
            f"SELECT CAST({dim_ident} AS STRING) AS x, {expr} AS y "
            f"FROM {logical_table_name} "
            f"WHERE {dim_ident} IS NOT NULL "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 30"
        )

    return f"SELECT 'All Data' AS x, {expr} AS y FROM {logical_table_name}"


def _kpi_value_sql_for_metric(metric_profile, logical_table_name="analysis_view"):
    profile = metric_profile if isinstance(metric_profile, dict) else {}
    expr = str(profile.get("expression") or "").strip() or "COUNT(*)"
    return f"SELECT {expr} AS value FROM {logical_table_name}"


def _kpi_trend_sql_for_metric(metric_profile, date_column, logical_table_name="analysis_view"):
    profile = metric_profile if isinstance(metric_profile, dict) else {}
    if not date_column:
        return ""
    date_ident = _quote_identifier(date_column)
    expr = str(profile.get("expression") or "").strip() or "COUNT(*)"
    return (
        f"SELECT CAST({date_ident} AS DATE) AS x, {expr} AS y "
        f"FROM {logical_table_name} "
        f"WHERE {date_ident} IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    )


def _extract_rca_titles(rca_list):
    titles = []
    seen = set()

    def _add(title):
        name = str(title or "").strip()
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        titles.append(name)

    if isinstance(rca_list, list):
        for item in rca_list:
            if isinstance(item, dict):
                _add(item.get("title") or item.get("name") or item.get("flow_name"))
            else:
                _add(item)
    elif isinstance(rca_list, dict):
        nested = rca_list.get("flows")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    _add(item.get("title") or item.get("name") or item.get("flow_name"))
                else:
                    _add(item)
        else:
            for key in rca_list.keys():
                _add(key)

    return titles


def _fetch_knowledge_base_from_db(module_name=None, logs=None):
    module_key = _resolve_kb_module_name(module_name)
    if not module_key:
        _log_kb_message(logs, "[KB] Knowledge base fetch failed: KB_MODULE_NAME is empty or not set in .env")
        return None
    now_ts = time.time()

    cached_item = _kb_cache.get(module_key)
    if cached_item:
        cached_data, fetched_at = cached_item
        age_seconds = int(max(0, now_ts - float(fetched_at or 0)))
        if cached_data is not None and age_seconds < KB_CACHE_TTL_SECONDS:
            _log_kb_message(logs, f"[KB] Using cached knowledge base (age: {age_seconds}s)")
            return cached_data

    _log_kb_message(logs, f"[KB] Fetching knowledge base from pgadmin_module for module: {module_key}")

    connection = None
    try:
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is not installed")

        db_name = str(os.getenv("DATA_DB_NAME") or "").strip()
        db_user = str(os.getenv("DATA_DB_USER") or "").strip()
        db_password = str(os.getenv("DATA_DB_PASSWORD") or "").strip()
        db_host = str(os.getenv("DATA_DB_HOST") or "").strip()
        db_port = str(os.getenv("DATA_DB_PORT") or "5432").strip() or "5432"
        db_sslmode = str(os.getenv("DATA_DB_SSLMODE") or "require").strip() or "require"

        if not all([db_name, db_user, db_password, db_host]):
            raise ValueError("Missing one or more required DATA_DB_* environment variables")

        connection = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            sslmode=db_sslmode,
        )
        with connection.cursor() as cursor:
            row = None
            try:
                cursor.execute(
                    """
                    SELECT knowledge_graph_data, metrics_data, extra_suggestions,
                           relationships, selected_columns, description, rca_list, tables, pos_tagging
                    FROM pgadmin_module
                    WHERE name = %s
                    """,
                    (module_key,),
                )
                row = cursor.fetchone()
            except Exception as e:
                if "pos_tagging" not in str(e).lower():
                    raise
                _log_kb_message(logs, "[KB] pos_tagging column missing in pgadmin_module; continuing without it")
                connection.rollback()
                cursor.execute(
                    """
                    SELECT knowledge_graph_data, metrics_data, extra_suggestions,
                           relationships, selected_columns, description, rca_list, tables
                    FROM pgadmin_module
                    WHERE name = %s
                    """,
                    (module_key,),
                )
                base_row = cursor.fetchone()
                if base_row is not None:
                    row = tuple(base_row) + ([],)

        if not row:
            raise LookupError(f"No knowledge base row found for module '{module_key}'")

        (
            knowledge_graph_raw,
            metrics_raw,
            extra_suggestions_raw,
            relationships_raw,
            selected_columns_raw,
            description_raw,
            rca_list_raw,
            tables_raw,
            pos_tagging_raw,
        ) = row

        kb_data = {
            "knowledge_graph_data": _parse_kb_json_field("knowledge_graph_data", knowledge_graph_raw),
            "metrics_data": _parse_kb_json_field("metrics_data", metrics_raw),
            "extra_suggestions": str(extra_suggestions_raw or "").strip(),
            "relationships": _parse_kb_json_field("relationships", relationships_raw),
            "selected_columns": _parse_kb_json_field("selected_columns", selected_columns_raw),
            "description": str(description_raw or "").strip(),
            "rca_list": _parse_kb_json_field("rca_list", rca_list_raw),
            "tables": _parse_kb_json_field("tables", tables_raw),
            "pos_tagging": _parse_kb_json_field("pos_tagging", pos_tagging_raw),
        }

        _kb_cache[module_key] = (kb_data, now_ts)

        table_count = len(_extract_table_names(kb_data.get("tables")))
        metric_count = len(_extract_metric_rows(kb_data.get("metrics_data")))
        rules_count = _count_numbered_rules(kb_data.get("extra_suggestions"))
        _log_kb_message(
            logs,
            f"[KB] Knowledge base fetched successfully — tables: {table_count}, metrics: {metric_count}, rules: {rules_count}",
        )
        return kb_data
    except Exception as e:
        _log_kb_message(logs, f"[KB] Knowledge base fetch failed: {str(e)}")
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _build_schema_context_from_knowledge_base(kb_data):
    if not isinstance(kb_data, dict):
        return ""

    knowledge_graph_data = kb_data.get("knowledge_graph_data")
    selected_columns = kb_data.get("selected_columns")
    relationships = kb_data.get("relationships")
    metrics_data = kb_data.get("metrics_data")
    tables_data = kb_data.get("tables")
    pos_tagging = kb_data.get("pos_tagging")

    selected_columns_map = _normalize_selected_columns_map(selected_columns)
    column_meta_map = _extract_knowledge_graph_column_meta(knowledge_graph_data)

    base_table = str(config_databricks_source_table() or "").strip()
    joinable_tails = _kb_joinable_table_tails(base_table, kb_data)

    if not selected_columns_map:
        for table_name in _extract_table_names(tables_data):
            table_key = str(table_name).lower()
            selected_columns_map[table_key] = {"table_name": table_name, "columns": []}

    section_two_lines = []
    for table_key, table_payload in selected_columns_map.items():
        table_name = table_payload.get("table_name") or table_key
        table_tail = _extract_table_tail(table_name) or str(table_name).lower()
        if joinable_tails and table_tail not in joinable_tails:
            continue
        table_meta = column_meta_map.get(str(table_key).lower(), {})
        columns = table_payload.get("columns") or []

        if not columns and table_meta:
            for col_key, meta in table_meta.items():
                columns.append(
                    {
                        "name": col_key,
                        "datatype": str((meta or {}).get("datatype") or "").strip(),
                        "description": str((meta or {}).get("description") or "").strip(),
                    }
                )

        formatted_columns = []
        for col in columns:
            col_name = str((col or {}).get("name") or "").strip()
            if not col_name:
                continue
            col_meta = table_meta.get(col_name.lower(), {}) if isinstance(table_meta, dict) else {}
            datatype = str((col or {}).get("datatype") or col_meta.get("datatype") or "unknown").strip()
            description_text = str((col or {}).get("description") or col_meta.get("description") or "No description").strip()
            formatted_columns.append(f"{col_name} ({datatype}) - {description_text}")

        if not formatted_columns:
            formatted_columns.append("No selected columns available")

        section_two_lines.append(f"TABLE: {table_name}")
        section_two_lines.append(f"COLUMNS: {', '.join(formatted_columns)}")

    relationship_groups = _extract_relationship_join_groups(relationships)
    if joinable_tails:
        relationship_groups = [
            rel for rel in relationship_groups
            if (
                (_extract_table_tail(rel.get("left_table")) in joinable_tails)
                and (_extract_table_tail(rel.get("right_table")) in joinable_tails)
            )
        ]
    metric_rows = _extract_metric_rows(metrics_data)
    table_names = []
    for t_name in _extract_table_names(tables_data):
        tail = _extract_table_tail(t_name) or str(t_name).lower()
        if joinable_tails and tail not in joinable_tails:
            continue
        table_names.append(t_name)

    section_one = "Section 1 - Tables:\n" + (
        ", ".join(table_names) if table_names else "No tables listed."
    )
    section_two = "Section 2 - Knowledge Graph Data (Selected Columns Only):\n" + (
        "\n".join(section_two_lines) if section_two_lines else "No knowledge-graph column guidance available."
    )

    section_three_lines = []
    for rel in relationship_groups:
        left_table = str(rel.get("left_table") or "").strip()
        right_table = str(rel.get("right_table") or "").strip()
        rel_type = str(rel.get("type") or "unknown").strip() or "unknown"
        keys = rel.get("keys") or []
        join_sql = str(rel.get("join_sql") or "No join keys").strip() or "No join keys"

        section_three_lines.append(
            f"RELATIONSHIP: {left_table} -> {right_table} (type: {rel_type})"
        )

        if len(keys) > 1:
            section_three_lines.append("JOIN ON (USE ALL KEYS TOGETHER):")
            for idx, pair in enumerate(keys):
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    l_col = str(pair[0] or "").strip()
                    r_col = str(pair[1] or "").strip()
                    cond = f"{left_table}.{l_col} = {right_table}.{r_col}"
                else:
                    cond = ""
                if not cond:
                    continue
                if idx == 0:
                    section_three_lines.append(cond)
                else:
                    section_three_lines.append(f"AND {cond}")
            section_three_lines.append(
                f"WARNING: This join requires ALL {len(keys)} keys. Missing any key produces incorrect results."
            )
        else:
            section_three_lines.append(
                f"JOIN ON (USE ALL KEYS TOGETHER): {join_sql}"
            )
    section_three = "Section 3 - Relationships (Grouped Multi-Key Joins):\n" + (
        "\n".join(section_three_lines) if section_three_lines else "No relationships available."
    )

    section_four_parts = []
    for metric in metric_rows:
        section_four_parts.append(f"METRIC: {metric.get('name') or 'Unnamed Metric'}")
        section_four_parts.append(f"FORMULA: {metric.get('formula') or 'No formula provided'}")
        section_four_parts.append(f"DESCRIPTION: {metric.get('description') or 'No description provided'}")
        section_four_parts.append("")
    section_four_body = "\n".join(section_four_parts).strip() or "No metrics data available."
    section_four = f"Section 4 - Metrics Data:\n{section_four_body}"

    if isinstance(pos_tagging, (dict, list)):
        pos_payload = json.dumps(pos_tagging, ensure_ascii=False)
    else:
        pos_payload = str(pos_tagging or "[]").strip() or "[]"
    section_five = f"Section 5 - POS Tagging:\n{pos_payload}"

    selected_rules_lines = []
    for table_key, payload in selected_columns_map.items():
        table_name = payload.get("table_name") or table_key
        cols = [str((c or {}).get("name") or "").strip() for c in (payload.get("columns") or [])]
        cols = [c for c in cols if c]
        selected_rules_lines.append(f"{table_name}: {', '.join(cols) if cols else '(none)'}")
    section_six = "Section 6 - Selected Columns (Strict Access Contract):\n" + (
        "\n".join(selected_rules_lines) if selected_rules_lines else "No selected columns available."
    )
    section_seven = (
        "Section 7 - SQL Access Rules:\n"
        "1. Use only selected columns listed in Section 6.\n"
        "2. Do not use columns outside Section 6 for any table.\n"
        "3. For grouped relationships, use all listed join keys together with AND.\n"
        "4. If a required column is not in selected columns, do not infer or use it."
    )

    return "\n\n".join(
        [section_one, section_two, section_three, section_four, section_five, section_six, section_seven]
    ).strip()


def _kb_context_char_budget():
    raw = str(os.getenv("KB_CONTEXT_MAX_CHARS") or str(KB_CONTEXT_MAX_CHARS_DEFAULT)).strip()
    try:
        return max(2500, min(50000, int(raw)))
    except Exception:
        return KB_CONTEXT_MAX_CHARS_DEFAULT


def _kb_extract_rules(extra_suggestions_text):
    lines = [ln.strip() for ln in str(extra_suggestions_text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    rules = []
    current = ""
    for ln in lines:
        if re.match(r"^\d+[\)\.\:\-]?\s+", ln):
            if current:
                rules.append(current.strip())
            current = ln
        elif current:
            current = f"{current} {ln}"
        else:
            rules.append(ln)
    if current:
        rules.append(current.strip())
    return rules


def _kb_intent_terms(text):
    raw_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", str(text or "").lower())
    stopwords = {
        "the", "and", "for", "with", "from", "into", "onto", "that", "this", "then",
        "than", "when", "what", "which", "where", "show", "give", "make", "build",
        "create", "chart", "kpi", "dashboard", "data", "query", "queries", "sql",
        "table", "tables", "column", "columns", "metric", "metrics", "value", "values",
        "using", "used", "need", "want", "please", "should", "must", "only",
    }
    terms = []
    seen = set()
    for token in raw_terms:
        if token in stopwords:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _kb_score_blob(blob_text, terms):
    text = str(blob_text or "").lower()
    if not text or not terms:
        return 0
    score = 0
    for term in terms:
        if term in text:
            score += 3 if len(term) >= 6 else 2
    return score


def _kb_metric_priority_index(metric_name):
    ordered = ["sales", "eco", "ulpo", "abv", "drr", "fill rate", "suspicious eco", "units"]
    name = str(metric_name or "").lower()
    for idx, key in enumerate(ordered):
        if key in name:
            return idx
    return len(ordered) + 1


def _truncate_text(text, max_len):
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len - 16)].rstrip() + " ...[truncated]"


def _join_sections_with_budget(sections, max_chars):
    built = []
    used = 0
    truncated = False
    for section in sections:
        block = str(section or "").strip()
        if not block:
            continue
        delim = 2 if built else 0
        add_len = len(block) + delim
        if used + add_len <= max_chars:
            built.append(block)
            used += add_len
            continue

        remain = max_chars - used - delim
        if remain > 140:
            built.append(_truncate_text(block, remain))
            used = max_chars
            truncated = True
        break
    return "\n\n".join(built).strip(), truncated


def _build_targeted_schema_context_from_knowledge_base(kb_data, intent_text="", max_chars=None, logs=None):
    if not isinstance(kb_data, dict):
        return ""

    budget = int(max_chars or _kb_context_char_budget())
    terms = _kb_intent_terms(intent_text)
    description = str(kb_data.get("description") or "").strip()

    selected_columns_map = _normalize_selected_columns_map(kb_data.get("selected_columns"))
    column_meta_map = _extract_knowledge_graph_column_meta(kb_data.get("knowledge_graph_data"))
    table_names = _extract_table_names(kb_data.get("tables"))
    relationship_lines = _extract_relationship_lines(kb_data.get("relationships"))
    metric_rows = _extract_metric_rows(kb_data.get("metrics_data"))
    rca_titles = _extract_rca_titles(kb_data.get("rca_list"))
    all_rules = _kb_extract_rules(kb_data.get("extra_suggestions"))

    metric_scored = []
    for row in metric_rows:
        metric_name = str((row or {}).get("name") or "").strip()
        blob = " ".join(
            [
                metric_name,
                str((row or {}).get("description") or ""),
                str((row or {}).get("formula") or ""),
            ]
        )
        score = _kb_score_blob(blob, terms)
        metric_scored.append((score, _kb_metric_priority_index(metric_name), metric_name.lower(), row))
    metric_scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    selected_metrics = [row for score, _, _, row in metric_scored if score > 0][:4]
    if not selected_metrics:
        selected_metrics = [row for _, _, _, row in metric_scored[:4]]

    selected_metric_names = [str((m or {}).get("name") or "").strip() for m in selected_metrics]
    selected_metric_formulas_text = " ".join(str((m or {}).get("formula") or "") for m in selected_metrics).lower()

    table_candidates = {}
    for table_name in table_names:
        key = str(table_name or "").strip().lower()
        if not key:
            continue
        table_candidates[key] = table_name
    for key, payload in (selected_columns_map or {}).items():
        table_name = str((payload or {}).get("table_name") or key).strip()
        if table_name:
            table_candidates[str(table_name).lower()] = table_name

    table_scored = []
    for key, table_name in table_candidates.items():
        score = _kb_score_blob(table_name, terms)
        if key and key in selected_metric_formulas_text:
            score += 4
        table_scored.append((score, table_name.lower(), table_name))
    table_scored.sort(key=lambda x: (-x[0], x[1]))

    selected_tables = [name for score, _, name in table_scored if score > 0][:4]
    if not selected_tables:
        selected_tables = [name for _, _, name in table_scored[:4]]
    if not selected_tables:
        selected_tables = table_names[:4]

    selected_table_keys = {str(t).lower() for t in selected_tables}

    selected_relationships = []
    for rel_line in relationship_lines:
        rel_low = str(rel_line).lower()
        if any(tbl in rel_low for tbl in selected_table_keys):
            selected_relationships.append(rel_line)
    if not selected_relationships:
        selected_relationships = relationship_lines[:8]
    else:
        selected_relationships = selected_relationships[:10]

    rule_scored = []
    metric_terms = _kb_intent_terms(" ".join(selected_metric_names))
    for idx, rule in enumerate(all_rules):
        score = _kb_score_blob(rule, terms) + _kb_score_blob(rule, metric_terms)
        if idx == 0:
            score += 1
        rule_scored.append((score, idx, rule))
    rule_scored.sort(key=lambda x: (-x[0], x[1]))
    selected_rules = [rule for score, _, rule in rule_scored if score > 0][:8]
    if not selected_rules:
        selected_rules = [rule for _, _, rule in rule_scored[:6]]

    rca_scored = []
    for idx, title in enumerate(rca_titles):
        score = _kb_score_blob(title, terms)
        rca_scored.append((score, idx, title))
    rca_scored.sort(key=lambda x: (-x[0], x[1]))
    selected_rca = [title for score, _, title in rca_scored if score > 0][:4]
    if not selected_rca:
        selected_rca = [title for _, _, title in rca_scored[:3]]

    section_two_lines = []
    for table_name in selected_tables:
        table_key = str(table_name).lower()
        table_payload = selected_columns_map.get(table_key, {"table_name": table_name, "columns": []})
        table_meta = column_meta_map.get(table_key, {})
        columns = list(table_payload.get("columns") or [])

        if not columns and table_meta:
            for col_key, meta in table_meta.items():
                columns.append(
                    {
                        "name": col_key,
                        "datatype": str((meta or {}).get("datatype") or "").strip(),
                        "description": str((meta or {}).get("description") or "").strip(),
                    }
                )

        column_scored = []
        for col in columns:
            col_name = str((col or {}).get("name") or "").strip()
            if not col_name:
                continue
            col_meta = table_meta.get(col_name.lower(), {}) if isinstance(table_meta, dict) else {}
            datatype = str((col or {}).get("datatype") or col_meta.get("datatype") or "unknown").strip()
            desc = str((col or {}).get("description") or col_meta.get("description") or "No description").strip()
            score = _kb_score_blob(f"{col_name} {desc}", terms)
            column_scored.append((score, col_name.lower(), f"{col_name} ({datatype}) - {desc}"))

        column_scored.sort(key=lambda x: (-x[0], x[1]))
        chosen_cols = [txt for score, _, txt in column_scored if score > 0][:10]
        if not chosen_cols:
            chosen_cols = [txt for _, _, txt in column_scored[:8]]
        if not chosen_cols:
            chosen_cols = ["No selected columns available"]

        section_two_lines.append(f"TABLE: {table_name}")
        section_two_lines.append(f"COLUMNS: {', '.join(chosen_cols)}")

    section_one = "Section 1 - Module Description:\n" + (
        _truncate_text(description or "No module description available.", 1800)
    )
    section_two = "Section 2 - Available Tables and Key Columns:\n" + (
        "\n".join(section_two_lines) if section_two_lines else "No table/column guidance available."
    )
    section_three = "Section 3 - Table Relationships:\n" + (
        "\n".join(selected_relationships) if selected_relationships else "No table relationships available."
    )

    section_four_parts = []
    for metric in selected_metrics:
        section_four_parts.append(f"METRIC: {metric.get('name') or 'Unnamed Metric'}")
        section_four_parts.append(f"FORMULA: {_truncate_text(metric.get('formula') or 'No formula provided', 700)}")
        section_four_parts.append(f"DESCRIPTION: {_truncate_text(metric.get('description') or 'No description provided', 420)}")
        section_four_parts.append("")
    section_four = "Section 4 - Metric Formulas:\n" + (
        "\n".join(section_four_parts).strip() or "No metric formulas available."
    )

    section_five = "Section 5 - Important Rules (extra_suggestions):\n" + (
        "\n".join(selected_rules) if selected_rules else "No extra rules provided."
    )
    section_six = "Section 6 - RCA Analysis Flows (brief):\n" + (
        "\n".join(selected_rca) if selected_rca else "No RCA analysis flows listed."
    )

    context_text, was_truncated = _join_sections_with_budget(
        [section_one, section_two, section_three, section_four, section_five, section_six],
        max_chars=budget,
    )

    _log_kb_message(
        logs,
        (
            f"[KB] Selective context applied (budget={budget} chars, terms={len(terms)}, "
            f"tables={len(selected_tables)}, metrics={len(selected_metrics)}, rules={len(selected_rules)})"
        ),
    )
    if was_truncated:
        _log_kb_message(logs, "[KB] Selective context truncated to respect KB context budget")

    return context_text


def _build_databricks_virtual_source(connection, include_sample_rows, logs=None, kb_data=None):
    base_table = _resolve_databricks_source_table(connection, logs=logs)
    base_columns_raw = _describe_databricks_table_columns(connection, base_table)
    base_columns, dropped_base_cols = _dedupe_schema_columns(base_columns_raw)
    if dropped_base_cols and logs is not None:
        logs.append(f"[SOURCE] Deduped {dropped_base_cols} duplicate base columns from {base_table}")
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
            kb_data=kb_data,
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
        "base_columns": base_columns,
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
        kb_data = None
        if _is_kb_enabled():
            resolved_kb_module = _resolve_kb_module_name()
            log.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=log)
        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=False,
            logs=log,
            kb_data=kb_data,
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
        _log_sql_text(log, "Filter Values", "sql", sql_text)
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


def _extract_table_tail(table_name):
    cleaned = _unquote_databricks_identifier(table_name)
    parts = [p.strip() for p in str(cleaned or "").split(".") if p and p.strip()]
    return parts[-1].lower() if parts else ""


def _build_available_column_lookup(available_columns):
    lookup = {}
    for col in (available_columns or []):
        col_name = str(col or "").strip()
        if not col_name:
            continue
        lookup[col_name.lower()] = col_name
    return lookup


def _resolve_analysis_view_alias_column(prefix, raw_column, available_lookup):
    prefix_s = str(prefix or "").strip().lower()
    raw_col = str(raw_column or "").strip()
    col_clean = clean_col_name(raw_col)
    if not prefix_s or not col_clean:
        return ""

    direct = f"{prefix_s}_{col_clean}"
    if direct.lower() in available_lookup:
        return available_lookup[direct.lower()]

    # Common typo/variant fallbacks in existing dimensional columns.
    if prefix_s == "product" and col_clean == "category":
        typo = "product_cateogry"
        if typo in available_lookup:
            return available_lookup[typo]

    hinted = ANALYSIS_VIEW_ALIAS_HINTS.get(col_clean)
    if hinted and hinted.lower().startswith(f"{prefix_s}_") and hinted.lower() in available_lookup:
        return available_lookup[hinted.lower()]

    prefix_values = [v for k, v in available_lookup.items() if k.startswith(f"{prefix_s}_")]
    if prefix_values:
        suffix_match = [v for v in prefix_values if v.lower().endswith(f"_{col_clean}")]
        if suffix_match:
            return sorted(suffix_match, key=len)[0]

        contains_match = [v for v in prefix_values if col_clean in v.lower()]
        if contains_match:
            return sorted(contains_match, key=len)[0]

    # For optional relationship sources (fill_rate / suspicious), emit a
    # deterministic alias even if the current default schema does not expose it.
    if prefix_s in {"fill_rate", "suspicious"}:
        return direct

    # If we do not know available columns, still emit deterministic alias.
    return direct if not available_lookup else ""


def _rewrite_dimension_qualified_refs(sql_text, qualifier_to_prefix, available_lookup):
    sql_value = str(sql_text or "")
    if not sql_value or not qualifier_to_prefix:
        return sql_value, []

    notes = []
    rewritten = sql_value

    for qualifier, prefix in sorted(qualifier_to_prefix.items(), key=lambda kv: len(kv[0]), reverse=True):
        qual = str(qualifier or "").strip()
        pref = str(prefix or "").strip().lower()
        if not qual or not pref:
            continue

        patt = re.compile(
            rf"(?i)(?<![A-Za-z0-9_`]){re.escape(qual)}\s*\.\s*`?([A-Za-z0-9_]+)`?"
        )

        replace_count = 0

        def _repl(m):
            nonlocal replace_count
            raw_col = str(m.group(1) or "").strip()
            mapped = _resolve_analysis_view_alias_column(pref, raw_col, available_lookup)
            if not mapped:
                return m.group(0)
            replace_count += 1
            return mapped

        rewritten = patt.sub(_repl, rewritten)
        if replace_count > 0:
            notes.append(f"Rewrote {replace_count} qualified column reference(s) for '{qual}' -> {pref}_*")

    return rewritten, notes


def _strip_known_physical_joins(sql_text):
    sql_value = str(sql_text or "")
    if not sql_value:
        return sql_value, {}, []

    join_patt = re.compile(
        r"(?is)\b(?:left|right|inner|full|cross)?\s*join\s+"
        r"(?P<table>[`A-Za-z0-9_.]+)\s*"
        r"(?:(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?\s+on\s+"
        r"(?P<cond>.+?)(?=\b(?:left|right|inner|full|cross)?\s*join\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\bunion\b|$)"
    )

    qualifiers = {}
    notes = []
    out = []
    cursor = 0

    for match in join_patt.finditer(sql_value):
        table_token = str(match.group("table") or "").strip()
        alias_token = str(match.group("alias") or "").strip()
        table_tail = _extract_table_tail(table_token)

        if table_tail in DATABRICKS_PHYSICAL_TABLE_TAILS and table_tail != DATABRICKS_LOGICAL_VIEW_NAME.lower():
            out.append(sql_value[cursor:match.start()])
            cursor = match.end()

            dim_prefix = DATABRICKS_DIM_PREFIX_BY_TABLE.get(table_tail)
            if dim_prefix:
                qualifiers[table_tail] = dim_prefix
                full_name = _unquote_databricks_identifier(table_token).lower()
                if full_name:
                    qualifiers[full_name] = dim_prefix
                if alias_token:
                    qualifiers[alias_token.lower()] = dim_prefix

            notes.append(f"Removed JOIN on physical table '{table_tail}'")

    out.append(sql_value[cursor:])
    rewritten = "".join(out)
    # Join removal can collapse keyword boundaries (e.g., "fact_invoiceGROUP BY").
    rewritten = re.sub(
        r"(?i)([A-Za-z0-9_`)\]])(?=(GROUP\s+BY|WHERE|ORDER\s+BY|HAVING|LIMIT|UNION)\b)",
        r"\1 ",
        rewritten,
    )
    return rewritten, qualifiers, notes


def _rewrite_unqualified_business_tokens(sql_text, available_lookup):
    sql_value = str(sql_text or "")
    if not sql_value or not available_lookup:
        return sql_value, []

    rewritten = sql_value
    notes = []

    for raw_token, mapped_alias in sorted(ANALYSIS_VIEW_ALIAS_HINTS.items(), key=lambda kv: len(kv[0]), reverse=True):
        source_key = str(raw_token or "").strip().lower()
        target_key = str(mapped_alias or "").strip().lower()
        if not source_key or not target_key:
            continue

        if source_key in available_lookup:
            continue
        target_name = available_lookup.get(target_key, "")
        if not target_name:
            target_prefix = target_key.split("_", 1)[0] if "_" in target_key else ""
            if target_prefix not in {"fill_rate", "suspicious", "retailer"}:
                continue
            target_name = str(mapped_alias)
        patt = re.compile(
            rf"(?i)(?<![A-Za-z0-9_`\.'])`?{re.escape(raw_token)}(?:`)?(?![A-Za-z0-9_`'])"
        )
        rewritten, count = patt.subn(target_name, rewritten)
        if count > 0:
            notes.append(f"Rewrote token '{raw_token}' to '{target_name}' ({count}x)")

    # Generic fallback: resolve snake_case / punctuation variants to the
    # single best available column by normalized token shape.
    normalized_candidates = {}
    for original_key, original_name in available_lookup.items():
        normalized_key = re.sub(r"[^a-z0-9]+", "", str(original_key or "").lower())
        if not normalized_key:
            continue
        normalized_candidates.setdefault(normalized_key, set()).add(str(original_name))

    token_patt = re.compile(
        r"(?i)(?<![A-Za-z0-9_`\.'])`?([A-Za-z_][A-Za-z0-9_]*)`?(?![A-Za-z0-9_`'])"
    )
    normalized_rewrites = 0

    def _normalized_repl(match_obj):
        nonlocal normalized_rewrites
        raw_token = str(match_obj.group(1) or "").strip()
        raw_lower = raw_token.lower()
        if not raw_lower:
            return match_obj.group(0)
        if raw_lower in SQL_RESERVED_WORDS:
            return match_obj.group(0)
        if raw_lower in available_lookup:
            return match_obj.group(0)

        normalized_key = re.sub(r"[^a-z0-9]+", "", raw_lower)
        if not normalized_key or normalized_key == raw_lower:
            return match_obj.group(0)

        candidates = normalized_candidates.get(normalized_key, set())
        if len(candidates) != 1:
            return match_obj.group(0)

        mapped_name = next(iter(candidates))
        if mapped_name.lower() == raw_lower:
            return match_obj.group(0)

        normalized_rewrites += 1
        return mapped_name

    rewritten = token_patt.sub(_normalized_repl, rewritten)
    if normalized_rewrites > 0:
        notes.append(f"Rewrote {normalized_rewrites} normalized token reference(s) to available columns")

    return rewritten, notes


def _enforce_analysis_view_sql_contract(sql_text, available_columns=None):
    sql_value = str(sql_text or "")
    if not sql_value:
        return sql_value, []

    notes = []
    available_lookup = _build_available_column_lookup(available_columns)
    rewritten = sql_value

    rewritten, join_qualifiers, join_notes = _strip_known_physical_joins(rewritten)
    notes.extend(join_notes)

    qualifier_map = dict(join_qualifiers)
    for table_name, prefix in DATABRICKS_DIM_PREFIX_BY_TABLE.items():
        qualifier_map[table_name] = prefix

    rewritten, qualified_notes = _rewrite_dimension_qualified_refs(rewritten, qualifier_map, available_lookup)
    notes.extend(qualified_notes)

    for table_tail in sorted(DATABRICKS_PHYSICAL_TABLE_TAILS, key=len, reverse=True):
        table_patt = re.compile(
            rf"(?i)\b(from|join)\s+((?:`?[A-Za-z0-9_]+`?\.){{0,2}}`?{re.escape(table_tail)}`?)\b"
        )
        rewritten, count = table_patt.subn(
            lambda m: f"{m.group(1)} {DATABRICKS_LOGICAL_VIEW_NAME}",
            rewritten,
        )
        if count > 0:
            notes.append(
                f"Replaced {count} physical-table FROM/JOIN reference(s) for '{table_tail}' with '{DATABRICKS_LOGICAL_VIEW_NAME}'"
            )

    rewritten, token_notes = _rewrite_unqualified_business_tokens(rewritten, available_lookup)
    notes.extend(token_notes)
    return rewritten, notes


def _normalize_databricks_sql_dialect(sql_text):
    sql_value = str(sql_text or "")

    # Databricks SQL expects STRING type; bare VARCHAR without length fails.
    sql_value = re.sub(r"(?i)\bAS\s+VARCHAR\s*(\(\s*\d+\s*\))?", "AS STRING", sql_value)
    sql_value = re.sub(r"(?i)::\s*VARCHAR\s*(\(\s*\d+\s*\))?", "::STRING", sql_value)

    return sql_value


def _split_top_level_csv(expr_text):
    parts = []
    buf = []
    depth = 0
    in_single = False
    in_double = False

    for ch in str(expr_text or ""):
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue
        if in_single or in_double:
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_select_alias_expression(sql_text, alias_name):
    alias = str(alias_name or "").strip().lower()
    sql_value = str(sql_text or "")
    if not alias or not sql_value:
        return ""

    select_match = re.search(r"(?is)^\s*select\s+(?P<select_body>.+?)\s+from\s+", sql_value)
    if not select_match:
        return ""

    select_fields = _split_top_level_csv(select_match.group("select_body"))
    for field in select_fields:
        text = str(field or "").strip()
        if not text:
            continue

        as_match = re.match(
            r"(?is)^(?P<expr>.+?)\s+as\s+`?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            text,
        )
        if as_match and str(as_match.group("alias") or "").strip().lower() == alias:
            return str(as_match.group("expr") or "").strip()

        tail_match = re.match(
            r"(?is)^(?P<expr>.+?)\s+`?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            text,
        )
        if tail_match and str(tail_match.group("alias") or "").strip().lower() == alias:
            return str(tail_match.group("expr") or "").strip()

    return ""


def _infer_custom_chart_xy_columns_from_sql(sql_text, schema_columns):
    cols = [str(c).strip() for c, _ in (schema_columns or []) if str(c).strip()]
    dtype_lookup = {str(c).lower(): str(t) for c, t in (schema_columns or [])}
    x_expr = _extract_select_alias_expression(sql_text, "x")
    y_expr = _extract_select_alias_expression(sql_text, "y")

    def _pick_source_col(expr_text, prefer_numeric=False):
        expr = str(expr_text or "").strip()
        if not expr or not cols:
            return ""
        refs = [str(c).strip() for c in _extract_sql_referenced_columns(expr, cols) if str(c).strip()]
        if not refs:
            return ""
        if prefer_numeric:
            for col in refs:
                if _is_numeric_dtype(dtype_lookup.get(str(col).lower(), "")):
                    return col
        return refs[0]

    return {
        "x_column": _pick_source_col(x_expr, prefer_numeric=False),
        "y_column": _pick_source_col(y_expr, prefer_numeric=True),
        "x_expression": x_expr,
        "y_expression": y_expr,
    }


def _rewrite_group_by_alias_mismatch(sql_text):
    sql_value = str(sql_text or "")
    if not sql_value:
        return sql_value, []

    select_match = re.search(r"(?is)^\s*select\s+(?P<select_body>.+?)\s+from\s+", sql_value)
    if not select_match:
        return sql_value, []

    group_match = re.search(r"(?is)\bgroup\s+by\s+(?P<group_body>.+?)(?=\border\s+by\b|\bhaving\b|\blimit\b|\bqualify\b|\bunion\b|$)", sql_value)
    if not group_match:
        return sql_value, []

    select_fields = _split_top_level_csv(select_match.group("select_body"))
    if len(select_fields) < 2:
        return sql_value, []

    dim_field = str(select_fields[0] or "").strip()
    alias_match = re.match(r"(?is)^(?P<expr>.+?)\s+as\s+`?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)`?\s*$", dim_field)
    if not alias_match:
        return sql_value, []

    dim_expr = str(alias_match.group("expr") or "").strip()
    dim_alias = str(alias_match.group("alias") or "").strip()
    if not dim_expr or not dim_alias:
        return sql_value, []

    if re.fullmatch(r"`?[A-Za-z_][A-Za-z0-9_]*`?", dim_expr):
        expr_name = str(dim_expr).strip("`").strip()
        if expr_name.lower() == dim_alias.lower():
            return sql_value, []
    else:
        return sql_value, []

    group_body = str(group_match.group("group_body") or "").strip()
    group_fields = _split_top_level_csv(group_body)
    if len(group_fields) != 1:
        return sql_value, []

    group_field = str(group_fields[0] or "").strip().strip("`")
    if group_field.lower() != dim_alias.lower():
        return sql_value, []

    start = group_match.start("group_body")
    end = group_match.end("group_body")
    rewritten = sql_value[:start] + "1 " + sql_value[end:]
    notes = [f"Rewrote GROUP BY {dim_alias} to GROUP BY 1 to align with selected dimension expression"]
    return rewritten, notes


def _select_mtd_anchor_date_column(sql_text, available_columns=None):
    sql_value = str(sql_text or "")
    columns = [str(c).strip() for c in (available_columns or []) if str(c).strip()]

    # Prefer the explicit date column used in MTD predicates when present.
    mtd_col_patterns = [
        re.compile(
            r'(?i)(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*>=\s*DATE_TRUNC\(\s*[\'"]month[\'"]\s*,\s*CURRENT_DATE(?:\(\s*\))?\s*\)'
        ),
        re.compile(
            r'(?i)(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*>\s*=\s*DATE_TRUNC\(\s*[\'"]month[\'"]\s*,\s*CURRENT_DATE(?:\(\s*\))?\s*\)'
        ),
    ]

    for patt in mtd_col_patterns:
        match = patt.search(sql_value)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip().strip("`")
        if not candidate:
            continue
        if columns:
            resolved = _find_col_case_insensitive(columns, [candidate])
            if resolved:
                return str(resolved)
        return candidate

    if columns:
        preferred = _find_col_case_insensitive(
            columns,
            [
                "BillDate",
                "bill_date",
                "InvoiceDate",
                "invoice_date",
                "OrderDate",
                "order_date",
                "Date",
                "date",
            ],
        )
        if preferred:
            return str(preferred)

        for col in columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                return str(col)

    return "BillDate"


def _rewrite_current_date_mtd_to_data_max(sql_text, available_columns=None, view_name=DATABRICKS_LOGICAL_VIEW_NAME):
    sql_value = str(sql_text or "")
    if not sql_value:
        return sql_value, []

    patt = re.compile(
        r'(?i)DATE_TRUNC\(\s*[\'"]month[\'"]\s*,\s*CURRENT_DATE(?:\(\s*\))?\s*\)'
    )
    if not patt.search(sql_value):
        return sql_value, []

    date_col = _select_mtd_anchor_date_column(sql_value, available_columns=available_columns)
    date_ident = _quote_identifier(date_col)
    anchor_expr = (
        f"COALESCE((SELECT MAX({date_ident}) FROM {view_name} WHERE {date_ident} IS NOT NULL), CURRENT_DATE())"
    )
    replacement = f"DATE_TRUNC('month', {anchor_expr})"
    rewritten, count = patt.subn(replacement, sql_value)

    notes = []
    if count > 0:
        notes.append(
            f"Rewrote {count} CURRENT_DATE month-bound reference(s) to data-driven anchor using '{date_col}'"
        )
    return rewritten, notes


def _query_uses_relationship_columns(sql_text, available_columns=None):
    text = str(sql_text or "")
    relationship_prefixes = {
        str(v or "").strip().lower()
        for v in DATABRICKS_DIM_PREFIX_BY_TABLE.values()
        if str(v or "").strip()
    }

    # Preferred path: decide from actually referenced input columns only.
    # This avoids false positives from output aliases (e.g. unique_retailer_count).
    if isinstance(available_columns, list) and available_columns:
        referenced = _extract_sql_referenced_columns(text, available_columns)
        for col in referenced:
            col_name = str(col or "").strip().lower()
            if not col_name:
                continue
            if any(col_name.startswith(f"{prefix}_") for prefix in relationship_prefixes):
                return True
        return False

    # Fallback path when available columns are not provided.
    for raw_token, mapped_alias in ANALYSIS_VIEW_ALIAS_HINTS.items():
        mapped = str(mapped_alias or "").strip().lower()
        if "_" not in mapped:
            continue
        mapped_prefix = mapped.split("_", 1)[0]
        if mapped_prefix not in relationship_prefixes:
            continue

        raw = str(raw_token or "").strip()
        mapped_raw = str(mapped_alias or "").strip()
        if not raw and not mapped_raw:
            continue

        raw_hit = False
        mapped_hit = False
        if raw:
            raw_hit = re.search(
                rf"(?i)(?<![A-Za-z0-9_`])`?{re.escape(raw)}`?(?![A-Za-z0-9_`])",
                text,
            ) is not None
        if mapped_raw:
            mapped_hit = re.search(
                rf"(?i)(?<![A-Za-z0-9_`])`?{re.escape(mapped_raw)}`?(?![A-Za-z0-9_`])",
                text,
            ) is not None

        if raw_hit or mapped_hit:
            return True

    return False


def _extract_sql_referenced_columns(sql_text, available_columns):
    sql = str(sql_text or "")
    if not sql or not available_columns:
        return set()

    matched = set()
    for col in available_columns:
        name = str(col or "").strip()
        if not name:
            continue
        esc = re.escape(name)
        if re.search(rf"(?i)(?<![A-Za-z0-9_`])`{esc}`(?![A-Za-z0-9_`])", sql):
            matched.add(name)
            continue
        if re.search(rf"(?i)(?<![A-Za-z0-9_`]){esc}(?![A-Za-z0-9_`])", sql):
            matched.add(name)

    # Also capture mapped aliases hinted by KB guard, even when the current
    # default schema snapshot does not yet include those columns.
    for mapped_alias in set(ANALYSIS_VIEW_ALIAS_HINTS.values()):
        alias_name = str(mapped_alias or "").strip()
        if not alias_name:
            continue
        esc_alias = re.escape(alias_name)
        if re.search(rf"(?i)(?<![A-Za-z0-9_`])`?{esc_alias}`?(?![A-Za-z0-9_`])", sql):
            matched.add(alias_name)
    return matched


def _resolve_manual_config_table_for_column(column_name, kb_data=None, default_table="analysis_view"):
    col = str(column_name or "").strip()
    if not col:
        return str(default_table or "analysis_view")

    col_l = col.lower()
    col_tail = col_l.split("_")[-1] if "_" in col_l else col_l
    default_tail = _extract_table_tail(default_table) or str(default_table or "analysis_view")

    selected_map = _normalize_selected_columns_map((kb_data or {}).get("selected_columns"))
    hits = []
    for _, payload in (selected_map or {}).items():
        t_name = str((payload or {}).get("table_name") or "").strip()
        if not t_name:
            continue
        cols = payload.get("columns") or []
        names = [str((c or {}).get("name") or "").strip().lower() for c in cols if str((c or {}).get("name") or "").strip()]
        if not names:
            continue
        if col_l in names:
            hits.append(t_name)
            continue
        if col_tail and any(n.split("_")[-1] == col_tail for n in names):
            hits.append(t_name)

    if len(hits) == 1:
        return hits[0]

    for table_tail, prefix in DATABRICKS_DIM_PREFIX_BY_TABLE.items():
        pfx = str(prefix or "").strip().lower()
        if pfx and col_l.startswith(f"{pfx}_"):
            return table_tail

    return default_tail


def _normalize_chart_manual_config_from_spec(chart_spec, default_table="analysis_view", default_chart_type="bar", kb_data=None):
    spec = chart_spec if isinstance(chart_spec, dict) else {}
    nested = spec.get("manual_config") if isinstance(spec.get("manual_config"), dict) else {}

    def _pick(*keys):
        for key in keys:
            value = nested.get(key) if key in nested else spec.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return ""

    agg = _pick("aggregation", "agg").upper()
    if agg not in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"}:
        agg = "AUTO"

    ctype = _pick("chart_type", "type").lower()
    if ctype not in ALLOWED_CUSTOM_CHART_TYPES:
        ctype = str(default_chart_type or "bar").strip().lower() or "bar"

    out = {
        "x_table": str(default_table or "analysis_view"),
        "x_column": _pick("x_column"),
        "y_table": str(default_table or "analysis_view"),
        "y_column": _pick("y_column"),
        "aggregation": agg or "AUTO",
        "chart_type": ctype,
        "metric_name": _pick("metric_name"),
        "metric_formula": _pick("metric_formula"),
    }

    if out.get("x_column"):
        out["x_table"] = _resolve_manual_config_table_for_column(
            out.get("x_column"),
            kb_data=kb_data,
            default_table=default_table,
        )
    if out.get("y_column"):
        out["y_table"] = _resolve_manual_config_table_for_column(
            out.get("y_column"),
            kb_data=kb_data,
            default_table=default_table,
        )

    metric_profiles = _build_kb_metric_profiles(kb_data, []) if isinstance(kb_data, dict) else []
    metric_profile = _resolve_metric_profile_for_hints(
        metric_profiles,
        metric_name=out.get("metric_name"),
        metric_formula=out.get("metric_formula"),
        preferred_columns=[out.get("y_column"), out.get("x_column")],
    )
    if metric_profile:
        if not out.get("metric_name"):
            out["metric_name"] = str(metric_profile.get("name") or "").strip()
        if not out.get("metric_formula"):
            out["metric_formula"] = str(metric_profile.get("formula") or "").strip()
        out["metric_unit"] = str(metric_profile.get("unit_type") or "").strip().lower()

        metric_axis = "y"
        metric_cols = [str(c).strip().lower() for c in (metric_profile.get("columns") or []) if str(c).strip()]
        metric_primary = str(metric_profile.get("primary_column") or "").strip().lower()
        x_col = str(out.get("x_column") or "").strip().lower()
        y_col = str(out.get("y_column") or "").strip().lower()
        if x_col and (x_col in metric_cols or (metric_primary and x_col == metric_primary)):
            metric_axis = "x"
        elif y_col and (y_col in metric_cols or (metric_primary and y_col == metric_primary)):
            metric_axis = "y"
        out["metric_axis"] = metric_axis
        out["dimension_axis"] = "x" if metric_axis == "y" else "y"
    else:
        metric_axis_hint = str(_pick("metric_axis") or "y").lower()
        out["metric_axis"] = "x" if metric_axis_hint == "x" else "y"
        out["dimension_axis"] = "x" if out.get("metric_axis") == "y" else "y"
        out["metric_unit"] = str(_pick("metric_unit") or "").strip().lower()

    return out


def _normalize_kpi_manual_config_from_spec(kpi_spec, default_table="analysis_view", kb_data=None):
    spec = kpi_spec if isinstance(kpi_spec, dict) else {}
    nested = spec.get("manual_config") if isinstance(spec.get("manual_config"), dict) else {}

    def _pick(*keys):
        for key in keys:
            value = nested.get(key) if key in nested else spec.get(key)
            text = str(value or "").strip()
            if text:
                return text
        return ""

    agg = _pick("aggregation", "agg").upper()
    if agg not in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"}:
        agg = "AUTO"

    col_name = _pick("column_name", "metric_column", "y_column", "x_column")
    resolved_table = str(default_table or "analysis_view")
    if col_name:
        resolved_table = _resolve_manual_config_table_for_column(
            col_name,
            kb_data=kb_data,
            default_table=default_table,
        )

    metric_profiles = _build_kb_metric_profiles(kb_data, []) if isinstance(kb_data, dict) else []
    metric_profile = _resolve_metric_profile_for_hints(
        metric_profiles,
        metric_name=_pick("metric_name", "label"),
        metric_formula=_pick("metric_formula"),
        preferred_columns=[col_name],
    )
    metric_name = _pick("metric_name", "label")
    metric_formula = _pick("metric_formula")
    if metric_profile:
        if not metric_name:
            metric_name = str(metric_profile.get("name") or "").strip()
        if not metric_formula:
            metric_formula = str(metric_profile.get("formula") or "").strip()
    metric_unit = _resolve_effective_metric_unit(
        metric_name=metric_name,
        label_text=_pick("label"),
        formula_text=metric_formula,
        expression_text=col_name,
        explicit_unit=_pick("metric_unit"),
        profile_unit=(metric_profile or {}).get("unit_type") if isinstance(metric_profile, dict) else "",
    )

    return {
        "table_name": str(resolved_table or default_table or "analysis_view"),
        "column_name": col_name,
        "aggregation": agg,
        "metric_name": metric_name,
        "metric_formula": metric_formula,
        "metric_unit": metric_unit,
    }


def _build_pruned_batch_query_source(connection, source_table, query_source, required_columns, logs=None, log_context="Batch"):
    req_cols = [str(c).strip() for c in (required_columns or []) if str(c).strip()]
    if not req_cols:
        return str(query_source or source_table).strip()

    fallback_source = str(query_source or source_table).strip()
    try:
        base_cols_raw = _describe_databricks_table_columns(connection, source_table)
        base_cols, dropped = _dedupe_schema_columns(base_cols_raw)
        if dropped and logs is not None:
            logs.append(f"[SOURCE] Deduped {dropped} duplicate base columns from {source_table} for batch prune")

        kb_data = None
        if _is_kb_enabled():
            kb_module = _resolve_kb_module_name()
            kb_data = _fetch_knowledge_base_from_db(module_name=kb_module, logs=logs)

        relation_model = _build_databricks_relationship_select_model(
            connection,
            source_table,
            base_cols,
            required_columns=req_cols,
            logs=logs,
            kb_data=kb_data,
        )
        if not relation_model:
            return fallback_source

        if logs is not None:
            logs.append(
                f"[PERF] {log_context} source pruned: columns={len(relation_model.get('schema_columns', []))}, "
                f"joins={len(relation_model.get('joined_tables', []))}"
            )
        return relation_model.get("query_source", fallback_source) or fallback_source
    except Exception as e:
        if logs is not None:
            logs.append(f"[WARN] Batch source prune failed; using fallback source: {str(e)}")
        return fallback_source


def _choose_effective_query_source(source_table, query_source, user_sql="", where_sql="", available_columns=None):
    source_base = str(source_table or "").strip()
    source_joined = str(query_source or "").strip()
    if not source_joined or source_joined == source_base:
        return source_base, False

    combined = f"{user_sql or ''}\n{where_sql or ''}"
    needs_join = _query_uses_relationship_columns(combined, available_columns=available_columns)
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


def _execute_databricks_user_sql(
    connection,
    user_sql,
    source_table,
    where_sql="",
    logs=None,
    context="Databricks Query",
    query_source=None,
    available_columns=None,
):
    normalized_sql = _normalize_databricks_sql_references(
        user_sql,
        source_table,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
    )
    contract_sql, contract_notes = _enforce_analysis_view_sql_contract(
        normalized_sql,
        available_columns=available_columns,
    )
    dialect_sql = _normalize_databricks_sql_dialect(contract_sql)
    agg_fixed_sql, agg_fix_notes = _rewrite_group_by_alias_mismatch(dialect_sql)
    mtd_rewritten_sql, mtd_notes = _rewrite_current_date_mtd_to_data_max(
        agg_fixed_sql,
        available_columns=available_columns,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
    )
    guarded_sql, notes = _apply_sql_security_and_cost_guardrails(mtd_rewritten_sql)
    if logs is not None:
        for note in contract_notes:
            logs.append(f"[KB-GUARD] {context}: {note}")
        if contract_sql != normalized_sql:
            logs.append(f"[KB-GUARD] {context}: Enforced analysis_view-only SQL contract")
        if dialect_sql != contract_sql:
            logs.append(f"[SECURITY] {context}: Normalized SQL types for Databricks dialect (VARCHAR -> STRING)")
        for note in agg_fix_notes:
            logs.append(f"[KB-GUARD] {context}: {note}")
        for note in mtd_notes:
            logs.append(f"[KB-GUARD] {context}: {note}")
        for note in notes:
            logs.append(f"[SECURITY] {context}: {note}")
        _log_sql_text(logs, context, "guarded_sql", guarded_sql)
        row_cap = _databricks_query_row_cap()
        if row_cap > 0:
            logs.append(f"[PERF] {context}: Applying Databricks row cap DATABRICKS_QUERY_ROW_CAP={row_cap}")

    effective_query_source, using_join_source = _choose_effective_query_source(
        source_table,
        query_source=query_source,
        user_sql=guarded_sql,
        where_sql=where_sql,
        available_columns=available_columns,
    )
    if logs is not None:
        if using_join_source:
            logs.append(f"[PERF] {context}: Using relationship-join source")
        elif query_source and str(query_source).strip() != str(source_table).strip():
            logs.append(f"[PERF] {context}: Using base fact source (join bypass)")

    if using_join_source and _safe_bool_env("DATABRICKS_PRUNE_SINGLE_SOURCE", True):
        required_columns = set()
        if isinstance(available_columns, list) and available_columns:
            required_columns.update(_extract_sql_referenced_columns(where_sql, available_columns))
            required_columns.update(_extract_sql_referenced_columns(guarded_sql, available_columns))
        if required_columns:
            effective_query_source = _build_pruned_batch_query_source(
                connection,
                source_table,
                effective_query_source,
                sorted(required_columns),
                logs=logs,
                log_context=f"{context} (single)",
            )
        elif logs is not None:
            logs.append(f"[PERF] {context}: Single-query prune skipped (no referenced columns resolved)")

    wrapped_sql = _wrap_sql_with_virtual_views(
        guarded_sql,
        source_table,
        where_sql=where_sql,
        view_name=DATABRICKS_LOGICAL_VIEW_NAME,
        query_source=effective_query_source,
    )
    _log_sql_text(logs, context, "wrapped_sql", wrapped_sql)

    df = fetch_dataframe(connection, wrapped_sql, readonly=True)

    if df.empty:
        return df, guarded_sql

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(0)
    return df, guarded_sql



def _execute_databricks_batch_widget_queries(connection, jobs, source_table, where_sql="", query_source=None, logs=None, context="Batch Query", available_columns=None):
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
        contract_sql, contract_notes = _enforce_analysis_view_sql_contract(
            normalized_sql,
            available_columns=available_columns,
        )
        dialect_sql = _normalize_databricks_sql_dialect(contract_sql)
        agg_fixed_sql, agg_fix_notes = _rewrite_group_by_alias_mismatch(dialect_sql)
        mtd_rewritten_sql, mtd_notes = _rewrite_current_date_mtd_to_data_max(
            agg_fixed_sql,
            available_columns=available_columns,
            view_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        guarded_sql, notes = _apply_sql_security_and_cost_guardrails(mtd_rewritten_sql)

        if logs is not None:
            for note in contract_notes:
                logs.append(f"[KB-GUARD] {ctx}: {note}")
            if contract_sql != normalized_sql:
                logs.append(f"[KB-GUARD] {ctx}: Enforced analysis_view-only SQL contract")
            if dialect_sql != contract_sql:
                logs.append(f"[SECURITY] {ctx}: Normalized SQL types for Databricks dialect (VARCHAR -> STRING)")
            for note in agg_fix_notes:
                logs.append(f"[KB-GUARD] {ctx}: {note}")
            for note in mtd_notes:
                logs.append(f"[KB-GUARD] {ctx}: {note}")
            for note in notes:
                logs.append(f"[SECURITY] {ctx}: {note}")

        sanitized_jobs.append({"key": key, "sql": guarded_sql, "context": ctx})
        _log_sql_text(logs, f"{context}:{key}", "job_sql", guarded_sql)

    if not sanitized_jobs:
        return {}

    any_job_needs_join = _query_uses_relationship_columns(where_sql, available_columns=available_columns)
    if not any_job_needs_join:
        any_job_needs_join = any(
            _query_uses_relationship_columns(job.get("sql", ""), available_columns=available_columns)
            for job in sanitized_jobs
        )

    base_source, using_join_source = _choose_effective_query_source(
        source_table,
        query_source=query_source,
        user_sql=("\n".join(job.get("sql", "") for job in sanitized_jobs) if any_job_needs_join else ""),
        where_sql=where_sql,
        available_columns=available_columns,
    )
    if logs is not None:
        if using_join_source:
            logs.append(f"[PERF] {context}: Using relationship-join source for batch")
        elif query_source and str(query_source).strip() != str(source_table).strip():
            logs.append(f"[PERF] {context}: Using base fact source for batch (join bypass)")

    if using_join_source and _safe_bool_env("DATABRICKS_PRUNE_BATCH_SOURCE", True):
        required_columns = set()
        if isinstance(available_columns, list) and available_columns:
            required_columns.update(_extract_sql_referenced_columns(where_sql, available_columns))
            for job in sanitized_jobs:
                required_columns.update(_extract_sql_referenced_columns(job.get("sql", ""), available_columns))

        if required_columns:
            base_source = _build_pruned_batch_query_source(
                connection,
                source_table,
                base_source,
                sorted(required_columns),
                logs=logs,
                log_context=context,
            )
        elif logs is not None:
            logs.append(f"[PERF] {context}: Batch prune skipped (no referenced columns resolved)")

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
    _log_sql_text(logs, context, "batch_sql", batch_sql)

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
    system_prompt = f"""
    You are a BI Expert.

    STEP 1: FILTERS -> EXACTLY 6 filter columns ranked by usefulness (best first).
    - First 3 are used as default visible filters.
    - First 6 are used as recommended/preloaded filters.
    - Prefer business dimensions suitable for interactive filtering (region, customer, product, channel, status, etc.).
    - Use exact schema column names only.
    STEP 2: KPIS -> EXACTLY 4 KPI SQL queries.
    - Each KPI must have: label, sql, trend_sql, metric_name, metric_formula, metric_unit, column_name, aggregation.
    - aggregation must be one of: SUM, COUNT, AVG, MIN, MAX, AUTO.
    - column_name must be the source metric column used for KPI value.
    - metric_name MUST be from Section 4 - Metrics Data only.
    - metric_formula MUST match the selected Section 4 metric (or a valid simplified expression of it), preserving key semantics like DISTINCT and numerator/denominator structure.
    - metric_unit must be one of: currency, currency_rate, count, percent, number.
    - KPI sql MUST implement the selected metric_formula semantics.
    STEP 3: CHARTS -> 4 SQL queries.
    - For EACH chart, also return manual axis mapping fields:
      x_column, y_column, aggregation, metric_name, metric_formula.
    - aggregation must be one of: SUM, COUNT, AVG, MIN, MAX, AUTO.
    - x_column and y_column must be the exact source column names used to build x and y in SQL.
    - chart metric_name MUST be from Section 4 - Metrics Data only.
    - Each non-map chart must be metric-vs-dimension (Power BI style):
      one axis is metric output, the other axis is a business dimension (product/item/region/etc.).
    - Prefer keeping metric on Y axis and dimension/date on X axis.

    SQL EXECUTION CONTRACT (STRICT):
    - The ONLY table allowed in generated SQL is {logical_table_name}.
    - NEVER reference physical/base tables such as fact_invoice, dim_product_master, dim_customer_master, fill_rate, final_invoice_with_material, or dim_retailer_master.
    - NEVER generate explicit JOINs to physical tables.
    - Use denormalized columns already available in {logical_table_name}, including prefixed relationship aliases such as product_* and customer_*.
    - When mapping metrics/dimensions from KB formulas, follow ONLY explicit relationships from Section 3; never invent join keys.

    CRITICAL DATA GRAIN RULE:
    - {logical_table_name} is transaction-level (one row per transaction/event).
    - Master entities (supplier/customer/product/employee etc.) can repeat across many rows.
    - Transaction metrics (revenue, quantity, totals, trends) can aggregate directly on {logical_table_name}.
    - Master attributes (rating, age, salary, static price, static score, etc.) MUST be deduplicated by entity key before AVG or similar stats.
    - For unique entity counts, use COUNT(DISTINCT <detected_entity_key>), not COUNT(*).

    MANDATORY CHART RULES:
    - Chart 0: Skip; it is reserved for India map and generated by backend.
    - Chart 1 (Line/Trend): MUST be a trend over time using a date/timestamp column.
    - Chart 2: Choose best visualization for the data (bar, pie, scatter, or line).
    - Charts 3-4: Choose the best visualization based on data and business value.
    - Never force heatmap. Use heatmap only when it genuinely adds value (two categorical + one numeric).
    - All four generated charts must be distinct in metric and dimension.
    - Use ONLY metrics from Section 4 for chart values and KPI values.
    - For date/time trend SQL (charts and KPI trend_sql), DO NOT add LIMIT; include the full selected date window.
    - For 12-month style trends, prefer month grain (for example DATE_TRUNC('month', date_col)).

    TITLE REQUIREMENTS:
    - Titles must clearly mention the metric and dimension.
    KPI REQUIREMENTS:
    - Return EXACTLY 4 KPIs in the kpis array.
    - Return ONLY JSON.
    """

    user_prompt = f"""
    Analyze '{logical_table_name}' using this data context:
    {master_schema_context}

    RETURN JSON:
    {{
        "filters": ["Region", "Status", "Category", "Customer_Name", "Product", "BillStatus"],
        "kpis": [
            {{ "label": "Average Bill Value (ABV)", "sql": "SELECT SUM(gross_value)/COUNT(DISTINCT bill_key) FROM {logical_table_name}", "trend_sql": "SELECT DATE_TRUNC('month', CAST(date_col AS DATE)) as x, SUM(gross_value)/COUNT(DISTINCT bill_key) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1", "metric_name": "ABV", "metric_formula": "SUM(gross_value)/COUNT(DISTINCT bill_key)", "metric_unit": "currency", "column_name": "gross_value", "aggregation": "AUTO" }},
            {{ "label": "Distinct Retailer Count (ECO)", "sql": "SELECT COUNT(DISTINCT retaileruid) FROM {logical_table_name}", "trend_sql": "SELECT DATE_TRUNC('month', CAST(date_col AS DATE)) as x, COUNT(DISTINCT retaileruid) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1", "metric_name": "ECO", "metric_formula": "COUNT(DISTINCT retaileruid)", "metric_unit": "count", "column_name": "retaileruid", "aggregation": "COUNT" }}
        ],
        "charts": [
            {{ "title": "Monthly Trend", "type": "line", "sql": "SELECT date_col as x, SUM(val) as y FROM {logical_table_name} GROUP BY 1 ORDER BY 1", "xlabel": "Date", "ylabel": "Amount", "metric_name": "ABV", "metric_formula": "SUM(val)", "x_column": "date_col", "y_column": "val", "aggregation": "SUM" }},
            {{ "title": "Top Categories by Value", "type": "bar", "sql": "SELECT CAST(cat1 AS VARCHAR) AS x, SUM(val) AS y FROM {logical_table_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 12", "xlabel": "Category", "ylabel": "Value", "metric_name": "ABV", "metric_formula": "SUM(val)", "x_column": "cat1", "y_column": "val", "aggregation": "SUM" }},
            {{ "title": "Region Mix by Metric", "type": "pie", "sql": "SELECT CAST(region AS VARCHAR) AS x, SUM(val) AS y FROM {logical_table_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 12", "xlabel": "Region", "ylabel": "Metric", "metric_name": "DRR", "metric_formula": "SUM(val)", "x_column": "region", "y_column": "val", "aggregation": "SUM" }}
        ]
    }}
    """

    # OLD (Option 4): split stable rules into system + user messages.
    # res, tokens = call_ai_with_retry([
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt},
    # ], json_mode=True, debug_logs=debug_logs, context="Generate Viz Config")
    prompt = f"{system_prompt}\n\n{user_prompt}"
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

    system_prompt = f"""
    You are a BI Expert.
    TASK:
    - Generate EXACTLY {needed} ADDITIONAL KPI specs.
    - These must be different from existing KPIs in metric meaning (not just renamed duplicates).

    RULES:
    1. Return ONLY JSON.
    2. SQL must be SELECT/WITH only and use only {logical_table_name}.
    3. Each KPI must include: label, sql, trend_sql, metric_name, metric_formula, metric_unit, column_name, aggregation.
    4. trend_sql must return aliases exactly x, y.
    5. For unique entity counts, use COUNT(DISTINCT entity_key), not COUNT(*).
    6. Avoid duplicate labels and duplicate SQL semantics.
    7. NEVER reference physical tables (fact_invoice, dim_product_master, dim_customer_master, fill_rate, final_invoice_with_material, dim_retailer_master).
    8. metric_name MUST be from Section 4 - Metrics Data only.
    9. Use ONLY Section 4 metrics for KPI values.
    10. Follow explicit relationship mappings from Section 3 in DATA CONTEXT when resolving metric columns; do not invent join keys.
    11. metric_unit must be one of: currency, currency_rate, count, percent, number.
    12. For date/time trend_sql, DO NOT add LIMIT; return full selected date window.
    """

    user_prompt = f"""
    We already have KPI specs for '{logical_table_name}':
    {existing_json}

    DATA CONTEXT:
    {master_schema_context}

    RETURN JSON:
    {{
      "kpis": [
        {{ "label": "KPI 1", "sql": "SELECT ... FROM {logical_table_name}", "trend_sql": "SELECT ... AS x, ... AS y FROM {logical_table_name} ...", "metric_name": "ABV", "metric_formula": "SUM(amount)", "metric_unit": "currency", "column_name": "amount", "aggregation": "SUM" }}
      ]
    }}
    """

    # OLD (Option 4): split stable rules into system + user messages.
    # res, tokens = call_ai_with_retry(
    #     [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     json_mode=True,
    #     debug_logs=debug_logs,
    #     context="Generate Additional KPIs",
    # )
    prompt = f"{system_prompt}\n\n{user_prompt}"
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
        cleaned[-1]["table_name"] = str(k.get("table_name") or "").strip()
        cleaned[-1]["column_name"] = str(k.get("column_name") or "").strip()
        cleaned[-1]["aggregation"] = str(k.get("aggregation") or "").strip().upper()
        cleaned[-1]["metric_name"] = str(k.get("metric_name") or label).strip()
        cleaned[-1]["metric_formula"] = str(k.get("metric_formula") or "").strip()
        cleaned[-1]["metric_unit"] = str(k.get("metric_unit") or "").strip().lower()
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

    for table_tail in DATABRICKS_PHYSICAL_TABLE_TAILS:
        if re.search(
            rf"(?i)(?<![A-Za-z0-9_`])(?:`?[A-Za-z0-9_]+`?\.){{0,2}}`?{re.escape(table_tail)}`?(?![A-Za-z0-9_`])",
            cleaned,
        ):
            return False

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False

    return True


def _relax_string_equality_predicates(sql_text):
    sql = str(sql_text or "")
    if not sql:
        return sql

    patt = re.compile(
        r"(?i)(?P<lhs>(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_\.]*))\s*=\s*'(?P<rhs>[^']*)'"
    )

    def _repl(m):
        lhs = str(m.group("lhs") or "").strip()
        rhs = str(m.group("rhs") or "")
        rhs_s = rhs.strip()
        if not lhs or not rhs_s:
            return m.group(0)

        # Keep likely numeric/date literals untouched.
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", rhs_s):
            return m.group(0)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", rhs_s):
            return m.group(0)
        if re.fullmatch(r"\d{4}/\d{2}/\d{2}", rhs_s):
            return m.group(0) 

        safe_rhs = rhs.replace("'", "''")
        return f"UPPER(TRIM(CAST({lhs} AS STRING))) = UPPER(TRIM('{safe_rhs}'))"

    return patt.sub(_repl, sql)

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


def _sql_generation_safety_rules_block(logical_view_name="analysis_view"):
    logical_view = str(logical_view_name or "analysis_view").strip() or "analysis_view"
    return f"""
SQL GENERATION SAFETY RULES (MANDATORY)

1. USE ONLY {logical_view}
- Do NOT use raw tables (fact_invoice, fill_rate, etc.)
- Do NOT write JOIN statements
- Assume all required columns already exist in {logical_view}

2. METRIC PRIORITY
- If a metric is provided -> ALWAYS use its formula
- NEVER replace metric with simple SUM/AVG of a column
- y_column is only a hint, NOT the final computation

3. GRAIN ALIGNMENT (CRITICAL)
- Metric must be computed at the SAME grain as the chart

If grouping by BillDate:
-> metric must be computed per BillDate (or finer grain first, then aggregated)

Examples:
- ULPO:
  Step 1: COUNT(DISTINCT Material_No) per (BillDate, RetailerUID)
  Step 2: AVG per BillDate

- DRR:
  DRR = total sales / number of days
  If grouped by BillDate:
    DO NOT divide by COUNT(DISTINCT BillDate)

4. NO GLOBAL METRIC REUSE
- NEVER compute a metric once and reuse across all dates
- Metrics must be recalculated per time bucket

5. AVOID DUPLICATION
- Do NOT join aggregated results back to raw data
- This causes duplication and incorrect results

6. VALID AGGREGATIONS
- Only use SUM/AVG on numeric columns
- Do NOT aggregate categorical fields (IDs, codes, names)

7. TIME SERIES RULES
- For date trends:
  -> GROUP BY date
  -> ORDER BY date ASC
  -> DO NOT sort by metric

8. VALIDATION BEFORE OUTPUT
Before returning SQL, check:
- Metric matches definition
- Grain aligns with GROUP BY
- No division-by-1 issue
- All columns exist in {logical_view}

If any check fails -> rewrite the query

9. SPECIAL METRIC RULE: SUSPICIOUS ECO

- Suspicious ECO MUST be computed as:

  COUNT(DISTINCT CASE
      WHEN suspicious_ret_rag_flag = 'R' THEN RetailerUID
  END)
  /
  COUNT(DISTINCT RetailerUID)

- ALWAYS use:
  suspicious_ret_rag_flag (NOT ret_rag_flag)

- NEVER:
  - use SUM(CASE WHEN ...) for this metric
  - use non-distinct counts
  - reference raw tables or write joins

- Assume suspicious_ret_rag_flag is already available in {logical_view}
""".strip()


def _sql_generation_final_checklist_block(logical_view_name="analysis_view"):
    logical_view = str(logical_view_name or "analysis_view").strip() or "analysis_view"
    return f"""
FINAL SQL CHECKLIST BEFORE OUTPUT
- Use only {logical_view}; no raw tables and no JOIN statements.
- If a metric/formula is provided, preserve its semantics exactly.
- Align metric grain to the chart/KPI time bucket; never reuse one global metric across all dates.
- Avoid division-by-1 patterns such as grouping by BillDate while dividing by COUNT(DISTINCT BillDate).
- For date trends, GROUP BY date and ORDER BY date ASC.
- For Suspicious ECO, use COUNT(DISTINCT CASE WHEN suspicious_ret_rag_flag = 'R' THEN RetailerUID END) / COUNT(DISTINCT RetailerUID).
- If any of these checks fail, rewrite the SQL before returning JSON.
""".strip()


def generate_custom_chart_plan(
    final_schema_context,
    user_prompt,
    debug_logs=None,
    table_name="final_view",
    metric_profiles=None,
    detected_metric=None,
    relationships_dict=None,
):
    profiles = metric_profiles if isinstance(metric_profiles, list) else []
    metric_lines = []
    for profile in profiles[:40]:
        if not isinstance(profile, dict):
            continue
        name = str(profile.get("name") or "").strip()
        formula = str(profile.get("formula") or profile.get("expression") or "").strip()
        unit_type = str(profile.get("unit_type") or "").strip().lower()
        if not name:
            continue
        unit_part = f" [{unit_type}]" if unit_type else ""
        metric_lines.append(f"- {name}{unit_part}: {formula or '(formula not provided)'}")
    metric_catalog_text = "\n".join(metric_lines) if metric_lines else "- (No KB Section 4 metrics available)"

    if metric_lines:
        metric_rules = """
        14. MEASURE RESTRICTION (KB Section 4):
           - Choose exactly one metric_name from the provided "KB SECTION 4 METRICS" list.
           - metric_name must exactly match one listed metric.
           - metric_formula must correspond to that selected KB metric (or a valid analysis_view-adapted expression preserving its semantics).
           - The chart measure in SQL (y/z) must be derived only from that selected KB metric.
           - Do NOT invent ad-hoc metrics outside KB Section 4.
           - Do NOT use COUNT(*) unless the selected KB metric explicitly requires it.
        """
    else:
        metric_rules = """
        14. KB Section 4 metrics were not available; infer the best metric from schema but still return metric_name and metric_formula.
        """

    metric_rules += """
    15. Include these fields in JSON: metric_name, metric_formula, x_column, y_column, aggregation.
       - x_column and y_column must be exact source column names used for x/y.
       - aggregation must be one of SUM, COUNT, AVG, MIN, MAX, AUTO.
    """

    system_prompt = """
    You are a senior BI analyst. Build exactly one chart spec from a natural language request.

    GOAL:
    Convert the request into one clear visualization query that matches user intent.

    INTENT PRIORITY (high to low):
    1. Metric intent (what to measure).
    2. Time intent (month/week/day/range if asked).
    3. Grouping intent (overall total vs split by dimension).
    4. Chart type intent (if user asks explicit type, follow it).

    RULES:
    1. Return ONLY JSON.
    2. Supported chart types: bar, line, scatter, pie, heatmap, table, treemap, waterfall.
    3. SQL must be a SELECT/WITH query using only final_view.
    3a. Do NOT reference physical/base tables (fact_invoice, dim_product_master, dim_customer_master, fill_rate, final_invoice_with_material, dim_retailer_master).
    3b. Do NOT create explicit JOINs to physical tables; use the denormalized columns already present in final_view (for example product_* and customer_* aliases).
    4. Alias output columns exactly:
       - bar/line/scatter/pie: x, y
       - heatmap: x, y, z
       - table: keep meaningful business column names (do NOT alias to generic columns/rows)
       - treemap: labels, parents, values
       - waterfall: x, y (optional measure)
    5. For category charts, include LIMIT 30 or less.
    5a. For date/time trend charts, DO NOT use LIMIT; include the full selected date window.
    6. Use safe casts where useful (CAST(col AS VARCHAR) for categories, CAST(date AS DATE) for dates).
    7. If the user explicitly asks for weeks/weekly on x-axis, use weekly grain (for example DATE_TRUNC('week', date_col)) and set xlabel to Week.
    8. If user asks for monthly/daily trends, prefer a real date/timestamp column with DATE_TRUNC over string date buckets.
    9. If user asks for an overall total trend, do NOT introduce extra grouping dimensions.
    10. If user asks to compare multiple named entities (for example A and B), include that comparison in SQL filters/grouping explicitly.
    11. final_view is transaction-level. For transactional metrics use final_view directly.
    12. For master/entity attributes (rating, age, salary, static price, scores, etc.), deduplicate by entity key before AVG-style aggregations.
    13. For unique entity questions, use COUNT(DISTINCT <detected_entity_key>) instead of COUNT(*).
    """ + metric_rules
    system_prompt = f"{system_prompt.rstrip()}\n\n{_sql_generation_safety_rules_block(table_name)}"

    metric_instruction_block = ""
    if isinstance(detected_metric, dict):
        metric_name = str(detected_metric.get("name") or "").strip()
        exact_formula_raw = detected_metric.get("formula")
        exact_formula = "" if exact_formula_raw is None else str(exact_formula_raw)
        if metric_name and exact_formula:
            metric_instruction_block = (
                f"CRITICAL METRIC INSTRUCTION: The user is asking about: {metric_name}. "
                f"You MUST use EXACTLY this formula as the basis for the Y axis metric. "
                f"Do NOT simplify, modify, or substitute this formula: {exact_formula}. "
                f"This formula may reference multiple tables. Keep ALL table references and ALL join conditions exactly as shown. "
                f"When adapting for a dimension (e.g. GROUP BY region), add the dimension to SELECT and GROUP BY but do NOT change the aggregation logic or remove any joins.\n\n"
            )

    join_rules_block = ""
    if isinstance(relationships_dict, dict) and relationships_dict:
        join_lines = [
            "CRITICAL JOIN RULES — FOLLOW EXACTLY: When joining these tables, you MUST use ALL join keys listed. Never join on fewer keys than shown. Use LEFT JOIN always."
        ]
        for rel_key, rel_payload in relationships_dict.items():
            if not isinstance(rel_payload, dict):
                continue
            if not isinstance(rel_key, (list, tuple)) or len(rel_key) < 2:
                continue
            left_table = str(rel_key[0] or "").strip()
            right_table = str(rel_key[1] or "").strip()
            if not left_table or not right_table:
                continue
            join_keys = rel_payload.get("join_keys") or []
            key_conditions = []
            for pair in join_keys:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    l_col = str(pair[0] or "").strip()
                    r_col = str(pair[1] or "").strip()
                    if l_col and r_col:
                        key_conditions.append(f"{left_table}.{l_col} = {right_table}.{r_col}")
            if not key_conditions:
                join_sql_text = str(rel_payload.get("join_sql") or "").strip()
                if join_sql_text:
                    key_conditions = [join_sql_text]
            if not key_conditions:
                continue
            key_count = len(key_conditions)
            if key_count > 1:
                join_lines.append(
                    f"{left_table} → {right_table}: USE ALL {key_count} KEYS TOGETHER: {' AND '.join(key_conditions)}"
                )
            else:
                join_lines.append(f"{left_table} → {right_table}: {key_conditions[0]}")
        if len(join_lines) > 1:
            join_rules_block = "\n".join(join_lines) + "\n\n"

    chart_structure_rule_block = (
        "CHART STRUCTURE RULE: Every chart must have: X axis — one dimension column aliased as x. "
        "Y axis — one metric aggregation from the METRIC FORMULAS above aliased as y. "
        "Never put two metrics on the same chart unless the user explicitly asks for comparison. "
        "The metric formula must be used exactly as specified above — do not simplify or approximate.\n\n"
    )

    user_prompt_text = f"""{metric_instruction_block}{chart_structure_rule_block}{join_rules_block}
    DATA SCHEMA (table: final_view):
    {final_schema_context}

    KB SECTION 4 METRICS (ONLY ALLOWED MEASURES):
    {metric_catalog_text}

    USER REQUEST:
    {user_prompt}

    {_sql_generation_final_checklist_block(table_name)}

    JSON format:
    {{
      "title": "Sales by Region",
      "type": "bar",
      "sql": "SELECT CAST(region AS VARCHAR) AS x, SUM(sales) AS y FROM final_view GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
      "xlabel": "Region",
      "ylabel": "Sales",
      "metric_name": "Sales",
      "metric_formula": "SUM(gross_value)",
      "x_column": "customer_region_name",
      "y_column": "gross_value",
      "aggregation": "SUM"
    }}
    """
    if table_name != "final_view":
        system_prompt = system_prompt.replace("final_view", table_name)
        user_prompt_text = user_prompt_text.replace("final_view", table_name)

    # OLD (Option 4): split stable rules into system + user messages.
    # res, tokens = call_ai_with_retry([
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt_text},
    # ], json_mode=True, debug_logs=debug_logs, context="Generate Custom Chart Plan")
    prompt = f"{system_prompt}\n\n{user_prompt_text}"
    if debug_logs is not None:
        debug_logs.append(f"[LLM PROMPT] Generate Custom Chart Plan\n{_redact_sensitive_text(prompt)}")
    res, tokens = call_ai_with_retry([
        {"role": "user", "content": prompt}
    ], json_mode=True, debug_logs=debug_logs, context="Generate Custom Chart Plan")
    try:
        parsed = json.loads(res) if res else None
    except Exception:
        parsed = None
    return parsed, tokens


def _enrich_custom_chart_plan_metric_metadata(ai_plan, metric_profiles, schema_columns, logs=None):
    plan = ai_plan if isinstance(ai_plan, dict) else {}
    profiles = metric_profiles if isinstance(metric_profiles, list) else []
    if not isinstance(plan, dict):
        return ai_plan

    sql_text = str(plan.get("sql") or "").strip()
    inferred = _infer_custom_chart_xy_columns_from_sql(sql_text, schema_columns)
    x_column_hint = str(plan.get("x_column") or "").strip() or str(inferred.get("x_column") or "").strip()
    y_column_hint = str(plan.get("y_column") or "").strip() or str(inferred.get("y_column") or "").strip()
    y_expr_hint = str(inferred.get("y_expression") or "").strip()

    if x_column_hint and not str(plan.get("x_column") or "").strip():
        plan["x_column"] = x_column_hint
    if y_column_hint and not str(plan.get("y_column") or "").strip():
        plan["y_column"] = y_column_hint

    agg_raw = str(plan.get("aggregation") or "").strip().upper()
    if agg_raw not in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"}:
        agg_raw = "AUTO"

    selected_profile = None
    if profiles:
        metric_name_hint = str(plan.get("metric_name") or "").strip().lower()
        if metric_name_hint:
            selected_profile = next(
                (p for p in profiles if str((p or {}).get("name") or "").strip().lower() == metric_name_hint),
                None,
            )

        if not selected_profile:
            metric_formula_hint = str(plan.get("metric_formula") or "").strip().lower()
            if metric_formula_hint:
                selected_profile = next(
                    (
                        p for p in profiles
                        if str((p or {}).get("formula") or "").strip().lower() == metric_formula_hint
                        or str((p or {}).get("expression") or "").strip().lower() == metric_formula_hint
                    ),
                    None,
                )

        if not selected_profile and y_column_hint:
            y_low = y_column_hint.lower()
            for profile in profiles:
                cols = [str(c).strip().lower() for c in ((profile or {}).get("columns") or []) if str(c).strip()]
                primary = str((profile or {}).get("primary_column") or "").strip().lower()
                if y_low and (y_low in cols or (primary and y_low == primary)):
                    selected_profile = profile
                    break

        if not selected_profile and y_expr_hint:
            for profile in profiles:
                expr = str((profile or {}).get("expression") or "").strip().lower()
                formula = str((profile or {}).get("formula") or "").strip().lower()
                y_expr_low = y_expr_hint.lower()
                if expr and expr in y_expr_low:
                    selected_profile = profile
                    break
                if formula and _metric_formula_looks_like_expression(formula) and formula in y_expr_low:
                    selected_profile = profile
                    break

        if not selected_profile and profiles:
            selected_profile = profiles[0]
            if logs is not None:
                logs.append(
                    "[KB-METRIC] Custom chart did not specify a valid KB metric; defaulted to first KB metric profile"
                )

    if selected_profile:
        metric_name = str((selected_profile or {}).get("name") or "").strip()
        metric_formula = str((selected_profile or {}).get("formula") or "").strip() or str(
            (selected_profile or {}).get("expression") or ""
        ).strip()
        if metric_name:
            plan["metric_name"] = metric_name
        if metric_formula:
            plan["metric_formula"] = metric_formula
        if agg_raw == "AUTO":
            inferred_agg = str((selected_profile or {}).get("aggregation") or "").strip().upper()
            if inferred_agg in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
                agg_raw = inferred_agg

    plan["aggregation"] = agg_raw or "AUTO"
    if y_column_hint:
        plan["y_column"] = y_column_hint
    if x_column_hint:
        plan["x_column"] = x_column_hint
    return plan


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



def _remove_limit_for_time_series_sql(sql_text, preferred_date_col=None):
    sql = str(sql_text or "").strip()
    if not sql:
        return sql, False

    lowered = sql.lower()
    if " as x" not in lowered:
        return sql, False

    is_time_series = False
    if re.search(r"(?is)date_trunc\s*\(", sql):
        is_time_series = True
    elif re.search(r"(?is)cast\s*\([^)]*\bas\s+date\s*\)\s+as\s+x\b", sql):
        is_time_series = True
    elif re.search(r"(?is)to_date\s*\([^)]*\)\s+as\s+x\b", sql):
        is_time_series = True
    elif preferred_date_col:
        col = re.escape(str(preferred_date_col).strip().strip("`"))
        if col and re.search(rf"(?is)(`?{col}`?)\s+as\s+x\b", sql):
            is_time_series = True

    if not is_time_series:
        return sql, False

    updated = re.sub(r"(?is)\s+limit\s+\d+\s*;?\s*$", "", sql).strip()
    return updated, updated != sql


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
    available_columns=None,
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
            available_columns=available_columns,
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
    available_map = {str(c).lower(): c for c, _ in schema_columns}
    normalized = {}

    def _add(keyword_raw, column_raw):
        keyword = str(keyword_raw or "").strip().lower()
        column_name = str(column_raw or "").strip()
        if not keyword or not column_name:
            return
        resolved = available_map.get(column_name.lower())
        if not resolved:
            return
        normalized[keyword] = resolved

    if isinstance(choice, dict):
        raw_choices = choice.get("choices")
        if isinstance(raw_choices, list):
            for item in raw_choices:
                if isinstance(item, dict):
                    _add(item.get("keyword"), item.get("column"))

        raw_resolved = choice.get("resolved_choices")
        if isinstance(raw_resolved, dict):
            for key, value in raw_resolved.items():
                _add(key, value)

        _add(choice.get("keyword"), choice.get("column"))
    elif isinstance(choice, list):
        for item in choice:
            if isinstance(item, dict):
                _add(item.get("keyword"), item.get("column"))

    return normalized or None


def _apply_custom_chart_clarification_to_prompt(user_prompt, clarification_choice, logs=None):
    if not clarification_choice:
        return user_prompt
    mappings = []
    if isinstance(clarification_choice, dict):
        single_keyword = str(clarification_choice.get("keyword") or "").strip()
        single_column = str(clarification_choice.get("column") or "").strip()
        if single_keyword and single_column:
            mappings.append((single_keyword, single_column))

        for key, value in clarification_choice.items():
            if key in {"keyword", "column", "choices", "resolved_choices"}:
                continue
            k = str(key or "").strip()
            v = str(value or "").strip()
            if k and v:
                mappings.append((k, v))

    if not mappings:
        return user_prompt

    deduped = []
    seen = set()
    for keyword, column in mappings:
        dedupe_key = (keyword.lower(), column.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append((keyword, column))
        if logs is not None:
            logs.append(f"[CLARIFY] User selected column '{column}' for keyword '{keyword}'")

    addition_lines = [f"- Interpret '{keyword}' using column '{column}'." for keyword, column in deduped]
    addition_lines.append("- Use these mappings consistently in SQL and labels.")
    addition = "\n\nUSER CLARIFICATIONS:\n" + "\n".join(addition_lines)
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

    chosen_map = {}
    if isinstance(clarification_choice, dict):
        single_keyword = str(clarification_choice.get("keyword") or "").strip().lower()
        single_column = str(clarification_choice.get("column") or "").strip()
        if single_keyword and single_column:
            chosen_map[single_keyword] = single_column

        for key, value in clarification_choice.items():
            if key in {"keyword", "column", "choices", "resolved_choices"}:
                continue
            k = str(key or "").strip().lower()
            v = str(value or "").strip()
            if k and v:
                chosen_map[k] = v

    columns_only = [c for c, _ in schema_columns]

    # Pass 1: keyword-to-column name ambiguity.
    for term in column_terms:
        term_l = str(term).strip().lower()
        if not term_l:
            continue
        # Ignore generic metric words; only clarify meaningful business terms.
        if term_l in CUSTOM_CHART_DISAMBIGUATION_STOPWORDS:
            continue
        if chosen_map.get(term_l):
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
        if chosen_map.get(term_l):
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


def _limit_pie_segments(x_values, y_values, top_n=PIE_TOP_LABEL_LIMIT, others_label="Others"):
    try:
        x_vals = list(x_values or [])
        y_vals = list(y_values or [])
    except Exception:
        return x_values, y_values, False

    size = min(len(x_vals), len(y_vals))
    if size <= 0:
        return x_values, y_values, False

    pairs = []
    for i in range(size):
        label = str(x_vals[i]).strip() or "Unknown"
        try:
            value = float(y_vals[i])
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        pairs.append((label, value))

    if len(pairs) <= max(1, int(top_n)):
        return [x for x, _ in pairs], [y for _, y in pairs], False

    pairs_sorted = sorted(pairs, key=lambda row: row[1], reverse=True)
    keep_n = max(1, int(top_n))
    kept = pairs_sorted[:keep_n]
    remaining_sum = float(sum(v for _, v in pairs_sorted[keep_n:]))

    final_x = [x for x, _ in kept]
    final_y = [y for _, y in kept]

    if abs(remaining_sum) > 0:
        label = others_label if others_label not in final_x else f"{others_label} (Grouped)"
        final_x.append(label)
        final_y.append(remaining_sum)

    return final_x, final_y, True


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


def _sanitize_sparkline_values(values, target_points=None, floor=0.0):
    out = []
    for v in (values or []):
        try:
            n = float(v)
        except Exception:
            continue
        if not np.isfinite(n):
            continue
        if n < floor:
            n = floor
        out.append(float(n))

    if target_points is not None:
        try:
            tp = int(target_points)
        except Exception:
            tp = 0
        if tp > 0 and len(out) > tp:
            out = out[-tp:]
    return out


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
    values = _sanitize_sparkline_values(values, target_points=target_points, floor=0.0)

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
            values = _sanitize_sparkline_values(alt_values, target_points=target_points, floor=0.0)
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
    # Final clamp: if majority is non-negative, clamp all to >= 0 to remove
    # partial-period artifacts.
    non_negative_count = sum(1 for v in values if v >= 0)
    if non_negative_count >= len(values) // 2:
        values = [max(0.0, float(v)) for v in values]

    return _sanitize_sparkline_values(values, target_points=target_points, floor=0.0)


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
            if chart_type == "pie":
                chart_data["x"], chart_data["y"], _ = _limit_pie_segments(
                    chart_data.get("x", []),
                    chart_data.get("y", []),
                    top_n=PIE_TOP_LABEL_LIMIT,
                )

    return chart_data

def generate_custom_kpi_plan(final_schema_context, user_prompt, debug_logs=None, table_name="final_view", detected_metric=None, relationships_dict=None):
    system_prompt = """
    You are a senior BI analyst. Build exactly one KPI spec from a natural language request.

    GOAL:
    Convert the request into one KPI value query plus one matching trend query.

    INTENT PRIORITY (high to low):
    1. Metric intent (what to measure).
    2. Filter/entity intent (which segment/items).
    3. Time intent (requested window/grain).

    RULES:
    1. Return ONLY JSON.
    2. SQL must be SELECT/WITH only and must use only final_view.
    2a. Do NOT reference physical/base tables (fact_invoice, dim_product_master, dim_customer_master, fill_rate, final_invoice_with_material, dim_retailer_master).
    2b. Do NOT generate explicit JOINs to physical tables; use denormalized final_view columns (for example product_* and customer_* aliases).
    3. Provide two queries:
       - value_sql: one-row KPI value query (single metric cell preferred).
       - trend_sql: time trend query with aliases exactly x, y for KPI sparkline.
    4. value_sql and trend_sql must represent the same KPI definition and filters.
    5. For trend_sql use a date/timestamp column when available and order by x.
    6. If the request asks for monthly/weekly/daily behavior, reflect that time grain in trend_sql.
    7. final_view is transaction-level. For transactional metrics aggregate directly.
    8. For master/entity attributes (rating, age, salary, static price, scores, etc.), deduplicate by entity key before AVG-style aggregations.
    9. For unique entity questions use COUNT(DISTINCT <detected_entity_key>) instead of COUNT(*).
    """
    system_prompt = f"{system_prompt.rstrip()}\n\n{_sql_generation_safety_rules_block(table_name)}"

    metric_instruction_block = ""
    if isinstance(detected_metric, dict):
        metric_name = str(detected_metric.get("name") or "").strip()
        exact_formula_raw = detected_metric.get("formula")
        exact_formula = "" if exact_formula_raw is None else str(exact_formula_raw)
        if metric_name and exact_formula:
            metric_instruction_block = (
                f"CRITICAL METRIC INSTRUCTION: The user is asking about: {metric_name}. "
                f"You MUST use EXACTLY this formula as the basis for the Y axis metric. "
                f"Do NOT simplify, modify, or substitute this formula: {exact_formula}. "
                f"This formula may reference multiple tables. Keep ALL table references and ALL join conditions exactly as shown. "
                f"When adapting for a dimension (e.g. GROUP BY region), add the dimension to SELECT and GROUP BY but do NOT change the aggregation logic or remove any joins.\n\n"
            )

    join_rules_block = ""
    if isinstance(relationships_dict, dict) and relationships_dict:
        join_lines = [
            "CRITICAL JOIN RULES — FOLLOW EXACTLY: When joining these tables, you MUST use ALL join keys listed. Never join on fewer keys than shown. Use LEFT JOIN always."
        ]
        for rel_key, rel_payload in relationships_dict.items():
            if not isinstance(rel_payload, dict):
                continue
            if not isinstance(rel_key, (list, tuple)) or len(rel_key) < 2:
                continue
            left_table = str(rel_key[0] or "").strip()
            right_table = str(rel_key[1] or "").strip()
            if not left_table or not right_table:
                continue
            join_keys = rel_payload.get("join_keys") or []
            key_conditions = []
            for pair in join_keys:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    l_col = str(pair[0] or "").strip()
                    r_col = str(pair[1] or "").strip()
                    if l_col and r_col:
                        key_conditions.append(f"{left_table}.{l_col} = {right_table}.{r_col}")
            if not key_conditions:
                join_sql_text = str(rel_payload.get("join_sql") or "").strip()
                if join_sql_text:
                    key_conditions = [join_sql_text]
            if not key_conditions:
                continue
            key_count = len(key_conditions)
            if key_count > 1:
                join_lines.append(
                    f"{left_table} → {right_table}: USE ALL {key_count} KEYS TOGETHER: {' AND '.join(key_conditions)}"
                )
            else:
                join_lines.append(f"{left_table} → {right_table}: {key_conditions[0]}")
        if len(join_lines) > 1:
            join_rules_block = "\n".join(join_lines) + "\n\n"

    user_prompt_text = f"""{metric_instruction_block}{join_rules_block}
    DATA SCHEMA (table: final_view):
    {final_schema_context}

    USER REQUEST:
    {user_prompt}

    {_sql_generation_final_checklist_block(table_name)}

    JSON format:
    {{
      "label": "Total Revenue",
      "value_sql": "SELECT SUM(revenue) AS value FROM final_view",
      "trend_sql": "SELECT CAST(order_date AS DATE) AS x, SUM(revenue) AS y FROM final_view GROUP BY 1 ORDER BY 1 LIMIT 24"
    }}
    """
    if table_name != "final_view":
        system_prompt = system_prompt.replace("final_view", table_name)
        user_prompt_text = user_prompt_text.replace("final_view", table_name)

    # OLD (Option 4): split stable rules into system + user messages.
    # res, tokens = call_ai_with_retry([
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt_text}
    # ], json_mode=True, debug_logs=debug_logs, context="Generate Custom KPI Plan")
    prompt = f"{system_prompt}\n\n{user_prompt_text}"
    if debug_logs is not None:
        debug_logs.append(f"[LLM PROMPT] Generate Custom KPI Plan\n{_redact_sensitive_text(prompt)}")
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


def _format_kpi_display_value(value, unit_type="number"):
    if value is None:
        return "0"

    try:
        if pd.isna(value):
            return "0"
    except Exception:
        pass

    try:
        n = float(value)
        unit = str(unit_type or "number").strip().lower()

        if unit == "percent":
            pct = n * 100.0 if abs(n) <= 1.5 else n
            return f"{pct:,.2f}%"

        if unit == "count":
            if abs(n) >= 10_000_000:
                return f"{(n / 10_000_000):,.2f} Cr"
            if abs(n) >= 100_000:
                return f"{(n / 100_000):,.2f} L"
            return f"{n:,.0f}" if float(n).is_integer() else f"{n:,.2f}"

        if unit in {"currency", "currency_rate"}:
            suffix = "/day" if unit == "currency_rate" else ""
            if abs(n) >= 10_000_000:
                return f"{(n / 10_000_000):,.2f} Cr{suffix}"
            if abs(n) >= 100_000:
                return f"{(n / 100_000):,.2f} L{suffix}"
            if abs(n) >= 1_000:
                return f"{n:,.0f}{suffix}"
            return (f"{n:,.2f}" if not float(n).is_integer() else f"{n:,.0f}") + suffix

        # Generic numeric formatting
        if abs(n) >= 1_000:
            return f"{n:,.0f}"
        if abs(n) >= 100 or float(n).is_integer():
            return f"{n:,.0f}"
        return f"{n:,.2f}"
    except Exception:
        return str(value)


def _build_custom_kpi_payload(plan, value_raw, sparkline_data):
    label = str((plan or {}).get("label") or "Custom KPI").strip() or "Custom KPI"
    unit_hint = _resolve_effective_metric_unit(
        metric_name=str((plan or {}).get("metric_name") or ""),
        label_text=label,
        description=str((plan or {}).get("description") or ""),
        formula_text=str((plan or {}).get("metric_formula") or ""),
        expression_text=str((plan or {}).get("sql") or ""),
        explicit_unit=str((plan or {}).get("metric_unit") or (plan or {}).get("unit_type") or ""),
    )
    if not sparkline_data:
        base = 0.0
        try:
            base = float(value_raw)
        except Exception:
            base = 0.0
        sparkline_data = [max(0.0, base)] * KPI_TREND_POINTS

    return {
        "label": label,
        "value": _format_kpi_display_value(value_raw, unit_type=unit_hint),
        "unit_type": unit_hint,
        "sparkline": _sanitize_sparkline_values(sparkline_data, target_points=KPI_TREND_POINTS, floor=0.0),
    }


def generate_custom_kpi_from_prompt_databricks(
    user_prompt,
    active_filters_json='{}',
    clarification_choice=None,
    date_range_override=None,
    allow_ambiguity_fallback=False,
    kb_module_name=None,
):
    connection = get_databricks_connection()
    llm_logs = []
    try:
        include_sample_rows = _llm_include_sample_rows()
        if not include_sample_rows:
            llm_logs.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

        kb_data = None
        resolved_kb_module = _resolve_kb_module_name(kb_module_name)
        if _is_kb_enabled():
            llm_logs.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=llm_logs)

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=llm_logs,
            kb_data=kb_data,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        base_columns = source_model.get("base_columns", [])
        base_column_names_lower = {str(c).strip().lower() for c, _ in (base_columns or [])}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        if _is_kb_enabled():
            if kb_data is not None:
                # OLD (Option 1): selective KB injection by intent.
                # kb_context = _build_targeted_schema_context_from_knowledge_base(
                #     kb_data,
                #     intent_text=f"custom kpi request {user_prompt}",
                #     logs=llm_logs,
                # )
                # if not kb_context:
                #     kb_context = _build_schema_context_from_knowledge_base(kb_data)

                # NEW: full KB context injection
                kb_context = _build_schema_context_from_knowledge_base(kb_data)
                if kb_context:
                    llm_logs.append("[KB] Using knowledge base context from pgadmin_module")
                    effective_schema_context = kb_context
                else:
                    llm_logs.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                    effective_schema_context = schema_context
            else:
                llm_logs.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                effective_schema_context = schema_context
        else:
            llm_logs.append("[KB] Disabled via KB_ENABLED=false, falling back to Databricks metadata")
            effective_schema_context = schema_context

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
        detected_metric = None
        relationships_dict = {}
        if isinstance(kb_data, dict):
            metrics_dict = _extract_metrics_from_kb(kb_data)
            relationships_dict = _extract_relationships_from_kb(kb_data)
            detected_metric = _detect_requested_metric(prompt_for_kpi, metrics_dict)
            if isinstance(detected_metric, dict):
                llm_logs.append(f"[KB METRIC] Detected metric: {str(detected_metric.get('name') or '').strip()}")

        ai_plan, tokens_used = generate_custom_kpi_plan(
            # OLD: using generic Databricks metadata
            # schema_context,
            effective_schema_context,
            prompt_for_kpi,
            debug_logs=llm_logs,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
            detected_metric=detected_metric,
            relationships_dict=relationships_dict,
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

        if isinstance(ai_plan, dict):
            value_sql_guard = str(ai_plan.get("value_sql") or ai_plan.get("sql") or "").strip()
            trend_sql_guard = str(ai_plan.get("trend_sql") or "").strip()
            combined_guard_sql = "\n".join([s for s in [value_sql_guard, trend_sql_guard] if s])
            if combined_guard_sql and not _validate_metric_formula_in_sql(combined_guard_sql, detected_metric):
                fallback_candidates = [
                    str(date_column or "").strip(),
                    str(_find_col_case_insensitive(column_names, [str(ai_plan.get("x_column") or "").strip()]) or "").strip(),
                    str(date_cols[0] if date_cols else "").strip(),
                    str(column_names[0] if column_names else "").strip(),
                ]
                fallback_dimension = next((c for c in fallback_candidates if c), "")
                fallback_trend_sql = _build_sql_from_metric_formula(detected_metric, fallback_dimension)
                if fallback_trend_sql:
                    ai_plan["trend_sql"] = fallback_trend_sql
                    ai_plan["value_sql"] = f"SELECT SUM(y) AS value FROM ({fallback_trend_sql}) AS metric_guard_src"
                    ai_plan["sql"] = ai_plan["value_sql"]

            ai_plan["value_sql"] = _validate_joins_in_sql(str(ai_plan.get("value_sql") or ai_plan.get("sql") or ""), relationships_dict)
            ai_plan["sql"] = ai_plan["value_sql"]
            ai_plan["trend_sql"] = _validate_joins_in_sql(str(ai_plan.get("trend_sql") or ""), relationships_dict)

        value_sql = str(ai_plan.get("value_sql") or ai_plan.get("sql") or "").strip()
        value_sql = value_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
        value_sql, value_contract_notes = _enforce_analysis_view_sql_contract(
            value_sql,
            available_columns=column_names,
        )
        for note in value_contract_notes:
            llm_logs.append(f"[KB-GUARD] Custom KPI Value: {note}")
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
            value_sql, value_contract_notes = _enforce_analysis_view_sql_contract(
                value_sql,
                available_columns=column_names,
            )
            for note in value_contract_notes:
                llm_logs.append(f"[KB-GUARD] Custom KPI Value (fallback): {note}")

        value_df, executed_value_sql = _execute_databricks_user_sql(
            connection,
            value_sql,
            source_table_base,
            query_source=source_table_query,
            where_sql=where_sql,
            available_columns=column_names,
            logs=llm_logs,
            context="Custom KPI Value",
        )
        value_raw = _extract_first_scalar(value_df, default=0.0)

        trend_sql = str(ai_plan.get("trend_sql") or "").strip()
        trend_sql = trend_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
        trend_sql, trend_contract_notes = _enforce_analysis_view_sql_contract(
            trend_sql,
            available_columns=column_names,
        )
        for note in trend_contract_notes:
            llm_logs.append(f"[KB-GUARD] Custom KPI Trend: {note}")
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
                        available_columns=column_names,
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
            sparkline_data = [max(0.0, base)] * KPI_TREND_POINTS

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
    kb_module_name=None,
):
    log = _LiveLogBuffer(context="Dashboard")
    total_tokens = 0
    log.append(f"[KB-GUARD] Active build: {KB_GUARD_BUILD_ID}")
    if not session_id:
        session_id = str(uuid.uuid4())
    generation_start = _step_start(log, "Dashboard Generation", f"session={session_id}")

    include_sample_rows = _llm_include_sample_rows()
    if not include_sample_rows:
        log.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

    connection = get_databricks_connection()
    try:
        kb_data = None
        resolved_kb_module = _resolve_kb_module_name(kb_module_name)
        if _is_kb_enabled():
            log.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=log)

        meta_step = _step_start(log, "Metadata Load")
        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=log,
            kb_data=kb_data,
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
        base_columns = source_model.get("base_columns", [])
        base_column_names_lower = {str(c).strip().lower() for c, _ in (base_columns or [])}
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        if _is_kb_enabled():
            if kb_data is not None:
                # OLD (Option 1): selective KB injection by intent.
                # dashboard_intent = (
                #     "dashboard overview filters and charts for sales, eco, ulpo, abv, drr, fill rate, suspicious eco, units"
                # )
                # kb_context = _build_targeted_schema_context_from_knowledge_base(
                #     kb_data,
                #     intent_text=dashboard_intent,
                #     logs=log,
                # )
                # if not kb_context:
                #     kb_context = _build_schema_context_from_knowledge_base(kb_data)

                # NEW: full KB context injection
                kb_context = _build_schema_context_from_knowledge_base(kb_data)
                if kb_context:
                    log.append("[KB] Using knowledge base context from pgadmin_module")
                    effective_schema_context = kb_context
                else:
                    log.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                    effective_schema_context = schema_context
            else:
                log.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                effective_schema_context = schema_context
        else:
            log.append("[KB] Disabled via KB_ENABLED=false, falling back to Databricks metadata")
            effective_schema_context = schema_context

        _step_done(
            log,
            "Metadata Load",
            meta_step,
            detail=f"columns={len(column_names)} date_cols={len(date_cols)} text_cols={len(text_cols)} numeric_cols={len(num_cols)}"
        )

        filter_step = _step_start(log, "Filter Parse + WHERE Build")
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
        _step_done(log, "Filter Parse + WHERE Build", filter_step, detail=f"filters_applied={filter_count}")

        date_range_step = _step_start(log, "Date Range Resolve")
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
            date_range_source = source_table_base if str(date_column).strip().lower() in base_column_names_lower else source_table_query
            try:
                range_df = fetch_dataframe(
                    connection,
                    f"SELECT MIN({date_ident}) AS min_date, MAX({date_ident}) AS max_date FROM {date_range_source} WHERE {date_ident} IS NOT NULL",
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
        _step_done(
            log,
            "Date Range Resolve",
            date_range_step,
            detail=f"selected={date_column or 'none'} cached={'yes' if cached_date_range_used else 'no'}"
        )

        if _is_databricks_mode_active() and STRICT_SQL_GUARDRAILS:
            log.append(
                f"[SECURITY] Databricks SQL guardrails active (SELECT/WITH only, forbidden keywords blocked, max LIMIT={AI_SQL_MAX_LIMIT})"
            )

        viz_step = _step_start(log, "LLM Viz Plan")
        plan, tokens = generate_viz_config(
            # OLD: using generic Databricks metadata
            # schema_context,
            effective_schema_context,
            debug_logs=log,
            logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )
        total_tokens += tokens
        _step_done(log, "LLM Viz Plan", viz_step, detail=f"tokens={tokens}")

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
                extra_kpi_step = _step_start(log, "LLM Additional KPI Plan", detail=f"needed={needed_kpis}")
                extra_kpis, extra_tokens = generate_additional_kpis(
                    # OLD: using generic Databricks metadata
                    # schema_context,
                    effective_schema_context,
                    plan_kpis,
                    needed_kpis,
                    debug_logs=log,
                    logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                )
                total_tokens += extra_tokens
                if extra_kpis:
                    plan["kpis"] = plan_kpis + extra_kpis
                _step_done(log, "LLM Additional KPI Plan", extra_kpi_step, detail=f"tokens={extra_tokens} returned={len(extra_kpis or [])}")

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
                    {
                        "title": "Overview",
                        "type": "line" if time_col else "bar",
                        "sql": fallback_sql,
                        "xlabel": "",
                        "ylabel": "",
                        "x_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                        "x_column": str(time_col or dim_col or ""),
                        "y_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                        "y_column": str(metric_col or ""),
                        "aggregation": "SUM" if metric_col else "COUNT",
                    }
                ],
            }

        metric_profiles = _build_kb_metric_profiles(kb_data, schema_columns)
        if metric_profiles and isinstance(plan, dict):
            log.append(f"[KB-METRIC] Enforcing metric-driven chart/KPI plan using {len(metric_profiles)} KB metric(s)")

            def _profile_matches_col(profile, col_name):
                col = str(col_name or "").strip().lower()
                if not col:
                    return False
                primary = str((profile or {}).get("primary_column") or "").strip().lower()
                if primary and col == primary:
                    return True
                cols = [str(c).strip().lower() for c in ((profile or {}).get("columns") or []) if str(c).strip()]
                if col in cols:
                    return True
                col_tail = col.split("_")[-1]
                return any(c.split("_")[-1] == col_tail for c in cols)

            profile_by_name = {
                str((p or {}).get("name") or "").strip().lower(): p
                for p in metric_profiles
                if str((p or {}).get("name") or "").strip()
            }

            normalized_kpis = []
            used_metric_names = set()
            for raw_kpi in (plan.get("kpis", []) or []):
                if not isinstance(raw_kpi, dict):
                    continue
                hint_name = str(raw_kpi.get("metric_name") or "").strip().lower()
                profile = profile_by_name.get(hint_name)
                if not profile:
                    profile = _resolve_metric_profile_for_hints(
                        metric_profiles,
                        metric_name=raw_kpi.get("metric_name"),
                        metric_formula=raw_kpi.get("metric_formula"),
                        preferred_columns=[raw_kpi.get("column_name"), raw_kpi.get("y_column"), raw_kpi.get("x_column")],
                    )
                if not profile:
                    continue
                p_name = str(profile.get("name") or "").strip()
                if not p_name:
                    continue

                kpi_spec = dict(raw_kpi)
                kpi_spec["metric_name"] = p_name
                kpi_spec["metric_formula"] = str(profile.get("formula") or kpi_spec.get("metric_formula") or "").strip()
                kpi_spec["metric_unit"] = str(profile.get("unit_type") or kpi_spec.get("metric_unit") or "").strip().lower()
                if not str(kpi_spec.get("column_name") or "").strip():
                    kpi_spec["column_name"] = str(
                        profile.get("primary_column")
                        or ((profile.get("columns") or [""])[0] if isinstance(profile.get("columns"), list) else "")
                        or ""
                    ).strip()
                agg = str(kpi_spec.get("aggregation") or profile.get("aggregation") or "AUTO").strip().upper()
                if agg not in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"}:
                    agg = str(profile.get("aggregation") or "AUTO").strip().upper()
                kpi_spec["aggregation"] = agg
                # Enforce KPI SQL from KB metric profile to preserve metric semantics (ABV/ECO/DRR/etc.).
                kpi_spec["sql"] = _kpi_value_sql_for_metric(
                    profile,
                    logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                )
                kpi_spec["trend_sql"] = _kpi_trend_sql_for_metric(
                    profile,
                    date_column,
                    logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                )
                normalized_kpis.append(kpi_spec)
                used_metric_names.add(p_name.lower())
                if len(normalized_kpis) >= 4:
                    break

            metric_idx = 0
            while len(normalized_kpis) < 4 and metric_profiles:
                profile = metric_profiles[metric_idx % len(metric_profiles)]
                metric_idx += 1
                p_name = str(profile.get("name") or f"Metric {metric_idx}").strip() or f"Metric {metric_idx}"
                duplicate = p_name.lower() in used_metric_names
                label_text = p_name if not duplicate else f"{p_name} KPI {len(normalized_kpis) + 1}"
                normalized_kpis.append(
                    {
                        "label": label_text,
                        "sql": _kpi_value_sql_for_metric(profile, logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME),
                        "trend_sql": _kpi_trend_sql_for_metric(profile, date_column, logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME),
                        "metric_name": p_name,
                        "metric_formula": str(profile.get("formula") or "").strip(),
                        "metric_unit": str(profile.get("unit_type") or "").strip().lower(),
                        "column_name": str(
                            profile.get("primary_column")
                            or ((profile.get("columns") or [""])[0] if isinstance(profile.get("columns"), list) else "")
                            or ""
                        ).strip(),
                        "aggregation": str(profile.get("aggregation") or "AUTO").strip().upper(),
                    }
                )
                used_metric_names.add(p_name.lower())

            dim_candidates = []
            seen_dims = set()
            for col in [date_column] + list(text_cols or []) + list(date_cols or []):
                c = str(col or "").strip()
                if not c:
                    continue
                ck = c.lower()
                if ck in seen_dims:
                    continue
                seen_dims.add(ck)
                dim_candidates.append(c)
            if not dim_candidates:
                for col, _ in schema_columns:
                    c = str(col or "").strip()
                    if not c:
                        continue
                    ck = c.lower()
                    if ck in seen_dims:
                        continue
                    seen_dims.add(ck)
                    dim_candidates.append(c)
                    if len(dim_candidates) >= 8:
                        break

            normalized_charts = []
            used_chart_pairs = set()
            for raw_chart in (plan.get("charts", []) or []):
                if not isinstance(raw_chart, dict):
                    continue
                hint_name = str(raw_chart.get("metric_name") or "").strip().lower()
                profile = profile_by_name.get(hint_name)
                if not profile:
                    profile = _resolve_metric_profile_for_hints(
                        metric_profiles,
                        metric_name=raw_chart.get("metric_name"),
                        metric_formula=raw_chart.get("metric_formula"),
                        preferred_columns=[raw_chart.get("y_column"), raw_chart.get("x_column")],
                    )
                if not profile:
                    continue

                x_col_hint = str(raw_chart.get("x_column") or "").strip()
                y_col_hint = str(raw_chart.get("y_column") or "").strip()
                metric_axis = "y"
                if _profile_matches_col(profile, x_col_hint):
                    metric_axis = "x"
                elif _profile_matches_col(profile, y_col_hint):
                    metric_axis = "y"

                dim_col = y_col_hint if metric_axis == "x" else x_col_hint
                if not dim_col or _profile_matches_col(profile, dim_col):
                    dim_col = next(
                        (d for d in dim_candidates if not _profile_matches_col(profile, d)),
                        dim_candidates[0] if dim_candidates else "",
                    )

                metric_col = str(profile.get("primary_column") or "").strip()
                if not metric_col:
                    metric_col = str(((profile.get("columns") or [""])[0]) or "").strip()
                x_col = metric_col if metric_axis == "x" else dim_col
                y_col = dim_col if metric_axis == "x" else metric_col
                if not x_col:
                    x_col = dim_col
                if not y_col:
                    y_col = metric_col

                ctype = str(raw_chart.get("type") or "").strip().lower()
                if ctype not in ALLOWED_CUSTOM_CHART_TYPES:
                    ctype = "line" if (date_column and str(dim_col).lower() == str(date_column).lower()) else "bar"

                pair_key = f"{str(profile.get('name') or '').lower()}::{str(dim_col).lower()}::{ctype}"
                if pair_key in used_chart_pairs:
                    continue
                used_chart_pairs.add(pair_key)

                metric_name = str(profile.get("name") or "Metric").strip() or "Metric"
                dim_title = str(dim_col or "Dimension").replace("_", " ").title()
                chart_title = str(raw_chart.get("title") or "").strip() or f"{metric_name} by {dim_title}"
                normalized_charts.append(
                    {
                        "title": chart_title,
                        "type": ctype,
                        "sql": _metric_dimension_sql(
                            profile,
                            dim_col,
                            ctype,
                            date_column=date_column,
                            logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                        ),
                        "xlabel": str(raw_chart.get("xlabel") or dim_title).strip(),
                        "ylabel": str(raw_chart.get("ylabel") or metric_name).strip(),
                        "x_column": x_col,
                        "y_column": y_col,
                        "aggregation": str(profile.get("aggregation") or "AUTO").strip().upper(),
                        "metric_name": metric_name,
                        "metric_formula": str(profile.get("formula") or "").strip(),
                        "metric_unit": str(profile.get("unit_type") or "").strip().lower(),
                        "metric_axis": metric_axis,
                    }
                )
                if len(normalized_charts) >= NON_MAP_CHART_TARGET:
                    break

            chart_type_cycle = ["line", "bar", "pie", "bar"]
            fill_idx = 0
            while len(normalized_charts) < NON_MAP_CHART_TARGET and metric_profiles:
                profile = metric_profiles[fill_idx % len(metric_profiles)]
                dim_col = dim_candidates[fill_idx % len(dim_candidates)] if dim_candidates else ""
                ctype = chart_type_cycle[len(normalized_charts) % len(chart_type_cycle)]
                if date_column and str(dim_col).lower() == str(date_column).lower():
                    ctype = "line"
                pair_key = f"{str(profile.get('name') or '').lower()}::{str(dim_col).lower()}::{ctype}"
                fill_idx += 1
                if pair_key in used_chart_pairs:
                    if fill_idx < 24:
                        continue
                used_chart_pairs.add(pair_key)

                metric_name = str(profile.get("name") or f"Metric {fill_idx}").strip() or f"Metric {fill_idx}"
                metric_col = str(profile.get("primary_column") or "").strip()
                if not metric_col:
                    metric_col = str(((profile.get("columns") or [""])[0]) or "").strip()
                normalized_charts.append(
                    {
                        "title": f"{metric_name} by {str(dim_col or 'Dimension').replace('_', ' ').title()}",
                        "type": ctype,
                        "sql": _metric_dimension_sql(
                            profile,
                            dim_col,
                            ctype,
                            date_column=date_column,
                            logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                        ),
                        "xlabel": str(dim_col or "Dimension").replace("_", " ").title(),
                        "ylabel": metric_name,
                        "x_column": dim_col,
                        "y_column": metric_col,
                        "aggregation": str(profile.get("aggregation") or "AUTO").strip().upper(),
                        "metric_name": metric_name,
                        "metric_formula": str(profile.get("formula") or "").strip(),
                        "metric_unit": str(profile.get("unit_type") or "").strip().lower(),
                        "metric_axis": "y",
                    }
                )

            if normalized_kpis:
                plan["kpis"] = normalized_kpis[:4]
            if normalized_charts:
                plan["charts"] = normalized_charts[:NON_MAP_CHART_TARGET]

        # Step 4 guardrail pass: validate/fix each planned chart using KB metric
        # formulas and KB relationship joins before execution. Skip silently when
        # KB data is unavailable.
        if isinstance(plan, dict) and isinstance(kb_data, dict):
            metrics_dict = _extract_metrics_from_kb(kb_data)
            relationships_dict = _extract_relationships_from_kb(kb_data)
            chart_items = plan.get("charts", [])
            if isinstance(chart_items, list):
                for chart_idx, chart_spec in enumerate(chart_items):
                    if not isinstance(chart_spec, dict):
                        continue

                    title_hint = str(chart_spec.get("title") or "").strip()
                    ylabel_hint = str(chart_spec.get("ylabel") or "").strip()
                    detect_hint = f"{ylabel_hint} {title_hint}".strip()
                    detected_metric = _detect_requested_metric(detect_hint, metrics_dict)

                    sql_current = str(chart_spec.get("sql") or "").strip()
                    if detected_metric and sql_current:
                        metric_ok = _validate_metric_formula_in_sql(sql_current, detected_metric)
                        if not metric_ok:
                            xlabel_hint = str(chart_spec.get("xlabel") or "").strip()
                            xcol_hint = str(chart_spec.get("x_column") or "").strip()
                            dim_candidates = [
                                xlabel_hint,
                                xlabel_hint.replace(" ", "_") if xlabel_hint else "",
                                xcol_hint,
                                str(date_column or "").strip(),
                                str(date_cols[0] if date_cols else "").strip(),
                                str(text_cols[0] if text_cols else "").strip(),
                            ]
                            dimension_col = ""
                            for cand in dim_candidates:
                                if not cand:
                                    continue
                                match = _find_col_case_insensitive(column_names, [cand])
                                if match:
                                    dimension_col = str(match).strip()
                                    break
                                if not dimension_col:
                                    dimension_col = str(cand).strip()

                            rebuilt_sql = _build_sql_from_metric_formula(
                                detected_metric,
                                dimension_col,
                            )
                            if rebuilt_sql:
                                log.append(
                                    f"[METRIC GUARD] Chart {chart_idx} rebuilt from KB formula for metric "
                                    f"{str((detected_metric or {}).get('name') or '').strip()}."
                                )
                                chart_spec["sql"] = rebuilt_sql

                    sql_for_join = str(chart_spec.get("sql") or "").strip()
                    if sql_for_join:
                        fixed_sql = _validate_joins_in_sql(sql_for_join, relationships_dict)
                        if fixed_sql != sql_for_join:
                            log.append(f"[JOIN FIX] Chart {chart_idx} SQL updated with missing KB join keys.")
                            chart_spec["sql"] = fixed_sql
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

        filter_def_step = _step_start(log, "Filter Definitions Build")
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
        _step_done(log, "Filter Definitions Build", filter_def_step, detail=f"filters={len(output.get('filters', []))} lazy={'yes' if lazy_filter_values else 'no'}")

        metric_profile_by_name = {
            str((m or {}).get("name") or "").strip().lower(): m
            for m in (metric_profiles or [])
            if str((m or {}).get("name") or "").strip()
        }

        def _resolve_kpi_unit_hint(label_text="", spec=None, manual_config=None):
            s = spec if isinstance(spec, dict) else {}
            mc = manual_config if isinstance(manual_config, dict) else {}

            metric_name = str(s.get("metric_name") or mc.get("metric_name") or "").strip()
            metric_formula = str(s.get("metric_formula") or mc.get("metric_formula") or "").strip()
            profile_unit = ""
            if metric_name:
                profile = metric_profile_by_name.get(metric_name.lower())
                if isinstance(profile, dict):
                    profile_unit = str(profile.get("unit_type") or "").strip().lower()

            return _resolve_effective_metric_unit(
                metric_name=metric_name,
                label_text=label_text,
                description=str(s.get("description") or ""),
                formula_text=metric_formula,
                expression_text=str(s.get("sql") or ""),
                explicit_unit=str(s.get("metric_unit") or mc.get("metric_unit") or ""),
                profile_unit=profile_unit,
            )

        def _run_kpi_spec(label, kpi_sql, trend_sql=None, context_prefix="KPI", manual_config=None, unit_hint=""):
            label_text = str(label or "Metric").strip() or "Metric"
            kpi_sql = str(kpi_sql or "").replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
            kpi_sql, kpi_contract_notes = _enforce_analysis_view_sql_contract(
                kpi_sql,
                available_columns=column_names,
            )
            for note in kpi_contract_notes:
                log.append(f"[KB-GUARD] {context_prefix} {label_text}: {note}")
            if not kpi_sql:
                raise ValueError("Empty KPI SQL")

            df, executed_value_sql = _execute_databricks_user_sql(
                connection,
                kpi_sql,
                source_table_base,
                query_source=source_table_query,
                where_sql=where_sql,
                available_columns=column_names,
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
                trend_sql, removed_ts_limit = _remove_limit_for_time_series_sql(trend_sql, preferred_date_col=date_column)
                if removed_ts_limit:
                    log.append(f"[TIME] {context_prefix} {label_text}: removed LIMIT from KPI trend SQL to cover full selected date window")
                trend_sql, trend_contract_notes = _enforce_analysis_view_sql_contract(
                    trend_sql,
                    available_columns=column_names,
                )
                for note in trend_contract_notes:
                    log.append(f"[KB-GUARD] {context_prefix} Trend {label_text}: {note}")
                if trend_sql:
                    try:
                        trend_df, executed_trend_sql = _execute_databricks_user_sql(
                            connection,
                            trend_sql,
                            source_table_base,
                            query_source=source_table_query,
                            where_sql=where_sql,
                            available_columns=column_names,
                            logs=log,
                            context=f"{context_prefix} Trend {label_text}",
                        )
                        if not trend_df.empty and trend_df.shape[1] >= 2:
                            sparkline_data = _extract_kpi_sparkline_from_df(trend_df, target_points=KPI_TREND_POINTS)
                            try:
                                kpi_val_num = float(val)
                                if kpi_val_num >= 0 and sparkline_data:
                                    sparkline_data = [max(0.0, float(v)) for v in sparkline_data]
                            except Exception:
                                pass
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
                "value": _format_kpi_display_value(val, unit_type=unit_hint),
                "unit_type": unit_hint,
                "sparkline": _sanitize_sparkline_values(sparkline_data, target_points=KPI_TREND_POINTS, floor=0.0),
                "sql": executed_value_sql,
                "trend_sql": executed_trend_sql or trend_sql or "",
                "manual_config": manual_config if isinstance(manual_config, dict) else {},
            }

        invoice_entity_col = _select_invoice_entity_key(schema_columns)
        if invoice_entity_col:
            log.append(f"[GUARD] Invoice entity key selected: {invoice_entity_col}")
        skip_row_count = _safe_bool_env("DATABRICKS_SKIP_FILTERED_ROW_COUNT", False)
        filtered_row_count = 0
        row_count_step = _step_start(log, "Filtered Row Count")
        if skip_row_count:
            filtered_row_count = 1  # Assume data exists to allow KPIs/Fallbacks to attempt execution
            log.append("[PERF] Filtered row count skipped via DATABRICKS_SKIP_FILTERED_ROW_COUNT")
            _step_done(log, "Filtered Row Count", row_count_step, detail="skipped")
        else:
            try:
                count_df = fetch_dataframe(
                    connection,
                    f"SELECT COUNT(*) AS c FROM {source_table_query}" + (f" WHERE {where_sql}" if where_sql else ""),
                    readonly=True,
                )
                if not count_df.empty:
                    filtered_row_count = int(count_df.iloc[0].get("c") or 0)
                _step_done(log, "Filtered Row Count", row_count_step, detail=f"rows={filtered_row_count}")
            except Exception as e:
                log.append(f"[WARN] Could not compute filtered row count for KPI fallback: {str(e)}")
                _step_done(log, "Filtered Row Count", row_count_step, detail="rows=unknown")

        def _enforce_invoice_count_kpi(kpi_obj):
            return _normalize_invoice_count_kpi_plan(
                kpi_obj,
                schema_columns,
                DATABRICKS_LOGICAL_VIEW_NAME,
                date_column=date_column,
                logs=log,
                entity_col=invoice_entity_col,
            )

        kpi_exec_step = _step_start(log, "KPI Execute")
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
                        manual_config=_normalize_kpi_manual_config_from_spec(
                            kpi,
                            default_table=source_table_base,
                            kb_data=kb_data,
                        ),
                        unit_hint=_resolve_kpi_unit_hint(
                            label_text=label,
                            spec=kpi,
                            manual_config=kpi.get("manual_config") if isinstance(kpi.get("manual_config"), dict) else {},
                        ),
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
        if metric_profiles:
            fallback_metric_col = next(
                (
                    str(
                        (m or {}).get("primary_column")
                        or ((((m or {}).get("columns") or [""])[0]) if isinstance((m or {}).get("columns"), list) else "")
                        or ""
                    ).strip()
                    for m in metric_profiles
                    if str(
                        (m or {}).get("primary_column")
                        or ((((m or {}).get("columns") or [""])[0]) if isinstance((m or {}).get("columns"), list) else "")
                        or ""
                    ).strip()
                ),
                None,
            )
        if not fallback_metric_col and num_cols:
            fallback_metric_col = num_cols[0]
            for candidate in preferred_numeric_names:
                matched = next((c for c in num_cols if str(c).lower() == candidate), None)
                if matched:
                    fallback_metric_col = matched
                    break

        fallback_kpis = []
        if metric_profiles:
            for profile in metric_profiles:
                metric_label = str((profile or {}).get("name") or "").strip()
                if not metric_label:
                    continue
                fallback_kpis.append(
                    {
                        "label": metric_label,
                        "sql": _kpi_value_sql_for_metric(profile, logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME),
                        "trend_sql": _kpi_trend_sql_for_metric(profile, date_column, logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME),
                        "metric_name": metric_label,
                        "metric_formula": str((profile or {}).get("formula") or "").strip(),
                        "metric_unit": str((profile or {}).get("unit_type") or "").strip().lower(),
                        "column_name": str(
                            (profile or {}).get("primary_column")
                            or ((((profile or {}).get("columns") or [""])[0]) if isinstance((profile or {}).get("columns"), list) else "")
                            or ""
                        ).strip(),
                        "aggregation": str((profile or {}).get("aggregation") or "AUTO").strip().upper(),
                    }
                )
                if len(fallback_kpis) >= 24:
                    break
        else:
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
            if not metric_profiles:
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
                            manual_config=_normalize_kpi_manual_config_from_spec(
                                fallback_kpi,
                                default_table=source_table_base,
                                kb_data=kb_data,
                            ),
                            unit_hint=_resolve_kpi_unit_hint(
                                label_text=label,
                                spec=fallback_kpi,
                                manual_config=fallback_kpi,
                            ),
                        )
                    )
                    used_labels.add(label_key)
                except Exception as e:
                    log.append(f"[WARN] KPI fallback failed ({label}): {str(e)}")

        while len(valid_kpis) < 4:
            default_metric_profile = metric_profiles[0] if metric_profiles else {}
            default_metric_col = str(
                (default_metric_profile or {}).get("primary_column")
                or ((((default_metric_profile or {}).get("columns") or [""])[0]) if isinstance((default_metric_profile or {}).get("columns"), list) else "")
                or ""
            ).strip()
            default_manual = {
                "table_name": _resolve_manual_config_table_for_column(
                    default_metric_col,
                    kb_data=kb_data,
                    default_table=source_table_base,
                ),
                "column_name": default_metric_col,
                "aggregation": str((default_metric_profile or {}).get("aggregation") or "AUTO").strip().upper(),
                "metric_name": str((default_metric_profile or {}).get("name") or "").strip(),
                "metric_formula": str((default_metric_profile or {}).get("formula") or "").strip(),
                "metric_unit": str((default_metric_profile or {}).get("unit_type") or "").strip().lower(),
            } if default_metric_profile else {}
            if filtered_row_count <= 0:
                valid_kpis.append({
                    "label": "No Data",
                    "value": "-",
                    "unit_type": str(default_manual.get("metric_unit") or "number").strip().lower() or "number",
                    "sparkline": [0.0] * KPI_TREND_POINTS,
                    "sql": "",
                    "trend_sql": "",
                    "manual_config": default_manual,
                })
            else:
                valid_kpis.append({
                    "label": f"Rows (Filtered) {len(valid_kpis)+1}",
                    "value": _format_kpi_display_value(filtered_row_count, unit_type="count"),
                    "unit_type": "count",
                    "sparkline": [float(filtered_row_count)] * KPI_TREND_POINTS,
                    "sql": "",
                    "trend_sql": "",
                    "manual_config": default_manual,
                })

        output["kpis"] = valid_kpis[:4]
        _step_done(log, "KPI Execute", kpi_exec_step, detail=f"output_kpis={len(output['kpis'])}")

        chart_exec_step = _step_start(log, "Chart Execute")
        charts = plan.get("charts", [])[:NON_MAP_CHART_TARGET]

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

        fallback_metric_profile = metric_profiles[0] if metric_profiles else None
        fallback_metric = str((fallback_metric_profile or {}).get("primary_column") or "").strip()
        if not fallback_metric and num_cols:
            fallback_metric = num_cols[0]
        fallback_metric_ident = _quote_identifier(fallback_metric) if fallback_metric else None

        def _apply_databricks_chart_fallback(c_data, chart_index, reason_label):
            for dim_col in fallback_dims:
                dim_ident = _quote_identifier(dim_col)

                if fallback_metric_profile:
                    fallback_sql = _metric_dimension_sql(
                        fallback_metric_profile,
                        dim_col,
                        str((c_data or {}).get("type") or "bar").lower(),
                        date_column=date_column,
                        logical_table_name=DATABRICKS_LOGICAL_VIEW_NAME,
                    )
                    fallback_title = (
                        f"{str((fallback_metric_profile or {}).get('name') or 'Metric')} by "
                        f"{dim_col.replace('_', ' ').title()}"
                    )
                    fallback_ylabel = str((fallback_metric_profile or {}).get("name") or "Metric")
                elif fallback_metric_ident:
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
                        available_columns=column_names,
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
            chart_manual_from_plan = _normalize_chart_manual_config_from_spec(
                chart,
                default_table=source_table_base,
                default_chart_type=chart_type,
                kb_data=kb_data,
            )

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
                "manual_config": chart_manual_from_plan,
            }

            chart_has_data = False
            failure_reason = None

            try:
                chart_sql = str(chart.get("sql", "")).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME)
                chart_sql, removed_ts_limit = _remove_limit_for_time_series_sql(
                    chart_sql,
                    preferred_date_col=date_column,
                )
                if removed_ts_limit:
                    log.append(f"[TIME] Chart {i}: removed LIMIT from time-series SQL to cover full selected date window")
                chart_sql, chart_contract_notes = _enforce_analysis_view_sql_contract(
                    chart_sql,
                    available_columns=column_names,
                )
                for note in chart_contract_notes:
                    log.append(f"[KB-GUARD] Chart {i}: {note}")
                df, executed_chart_sql = _execute_databricks_user_sql(
                    connection,
                    chart_sql,
                    source_table_base,
                    query_source=source_table_query,
                    where_sql=where_sql,
                    available_columns=column_names,
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
                                    available_columns=column_names,
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
                            if chart_type == "pie":
                                c_data["x"], c_data["y"], pie_limited = _limit_pie_segments(
                                    c_data.get("x", []),
                                    c_data.get("y", []),
                                    top_n=PIE_TOP_LABEL_LIMIT,
                                )
                                if pie_limited:
                                    log.append(
                                        f"[INFO] Chart {i} pie labels limited to top {PIE_TOP_LABEL_LIMIT} with Others bucket"
                                    )

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

        while len(output["charts"]) < NON_MAP_CHART_TARGET:
            idx = len(output["charts"])
            default_metric_profile = metric_profiles[0] if metric_profiles else {}
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
                "manual_config": {
                    "x_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                    "x_column": "",
                    "y_table": _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                    "y_column": "",
                    "aggregation": "AUTO",
                    "chart_type": "bar",
                    "metric_name": str((default_metric_profile or {}).get("name") or "").strip(),
                    "metric_formula": str((default_metric_profile or {}).get("formula") or "").strip(),
                    "metric_unit": str((default_metric_profile or {}).get("unit_type") or "").strip().lower(),
                    "metric_axis": "y",
                    "dimension_axis": "x",
                },
            })

        map_chart = _build_map_chart_from_databricks(
            connection=connection,
            source_table_base=source_table_base,
            source_table_query=source_table_query,
            where_sql=where_sql,
            column_names=column_names,
            schema_columns=schema_columns,
            num_cols=num_cols,
            metric_profiles=metric_profiles,
            kb_data=kb_data,
            schema_context=effective_schema_context,
            logs=log,
        )
        output["charts"].insert(0, map_chart)
        for i, c in enumerate(output["charts"]):
            c["id"] = f"chart_{i}"
        output["charts"] = output["charts"][:TOTAL_CHART_TARGET]
        _step_done(log, "Chart Execute", chart_exec_step, detail=f"output_charts={len(output['charts'])}")

        output["__cache"] = {
            "filters": output.get("filters", []),
            "source_table": source_table_base,
            "date_range": output.get("date_range"),
            "selected_date_column": output.get("selected_date_column"),
        }
        _step_done(
            log,
            "Dashboard Generation",
            generation_start,
            detail=f"tokens={total_tokens} charts={len(output.get('charts', []))} kpis={len(output.get('kpis', []))}"
        )
        return output
    except Exception as e:
        _step_fail(log, "Dashboard Generation", generation_start, e)
        try:
            setattr(e, "_dashboard_logs", list(log))
        except Exception:
            pass
        raise
    finally:
        connection.close()





def execute_dashboard_filter_refresh_databricks(
    active_filters_json=None,
    widget_state=None,
    session_id=None,
    filters_override=None,
    date_range_override=None,
):
    log = _LiveLogBuffer(context="FilterRefresh")
    if not session_id:
        session_id = str(uuid.uuid4())

    widget_state = widget_state if isinstance(widget_state, dict) else {}
    kpi_specs = widget_state.get("kpis", [])
    if not isinstance(kpi_specs, list):
        kpi_specs = []
    chart_specs_raw = widget_state.get("charts", [])
    chart_specs = []
    map_metric_name_hint = ""
    map_metric_formula_hint = ""
    if isinstance(chart_specs_raw, list):
        for spec in chart_specs_raw:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("type") or "").strip().lower() == "india_map":
                map_manual = spec.get("manual_config") if isinstance(spec.get("manual_config"), dict) else {}
                map_metric_name_hint = str(
                    map_manual.get("metric_name")
                    or spec.get("metric_name")
                    or map_metric_name_hint
                    or ""
                ).strip()
                map_metric_formula_hint = str(
                    map_manual.get("metric_formula")
                    or spec.get("metric_formula")
                    or map_metric_formula_hint
                    or ""
                ).strip()
                continue
            chart_specs.append(spec)
            if len(chart_specs) >= NON_MAP_CHART_TARGET:
                break
    if not isinstance(chart_specs, list):
        chart_specs = []

    connection = get_databricks_connection()
    try:
        kb_data = None
        if _is_kb_enabled():
            resolved_kb_module = _resolve_kb_module_name()
            log.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=log)

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=False,
            logs=log,
            kb_data=kb_data,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_columns = source_model["schema_columns"]
        schema_context = source_model.get("schema_context") or ""
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        text_cols = [c for c, t in schema_columns if _is_text_dtype(t)]
        num_cols = [c for c, t in schema_columns if _is_numeric_dtype(t)]
        base_columns = source_model.get("base_columns", [])
        base_column_names_lower = {str(c).strip().lower() for c, _ in (base_columns or [])}
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
            date_range_source = source_table_base if str(date_column).strip().lower() in base_column_names_lower else source_table_query
            try:
                range_df = fetch_dataframe(
                    connection,
                    f"SELECT MIN({date_ident}) AS min_date, MAX({date_ident}) AS max_date FROM {date_range_source} WHERE {date_ident} IS NOT NULL",
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
                    "sparkline": _sanitize_sparkline_values(raw_spark, target_points=KPI_TREND_POINTS, floor=0.0),
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
                "manual_config": spec.get("manual_config") if isinstance(spec.get("manual_config"), dict) else {},
            })

        chart_runtime = []
        for idx, spec in enumerate(chart_specs):
            if not isinstance(spec, dict):
                continue
            chart_id = str(spec.get("id") or f"chart_{idx}")
            chart_type = str(spec.get("type") or "bar").lower()
            if chart_type not in ALLOWED_CUSTOM_CHART_TYPES:
                chart_type = "bar"
            chart_manual_from_spec = _normalize_chart_manual_config_from_spec(
                spec,
                default_table=source_table_base,
                default_chart_type=chart_type,
                kb_data=kb_data,
            )
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
                    "manual_config": chart_manual_from_spec,
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
                "manual_config": chart_manual_from_spec,
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
                    available_columns=column_names,
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
                    "manual_config": item.get("manual_config") if isinstance(item.get("manual_config"), dict) else {},
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
                sparkline_data = [max(0.0, base)] * KPI_TREND_POINTS

            item_manual = item.get("manual_config") if isinstance(item.get("manual_config"), dict) else {}
            effective_unit = _resolve_effective_metric_unit(
                metric_name=str((item_manual or {}).get("metric_name") or ""),
                label_text=label,
                formula_text=str((item_manual or {}).get("metric_formula") or ""),
                explicit_unit=str((item_manual or {}).get("metric_unit") or ""),
            )

            output["kpis"].append({
                "label": label,
                "value": _format_kpi_display_value(
                    value_raw,
                    unit_type=effective_unit,
                ),
                "unit_type": effective_unit,
                "sparkline": _sanitize_sparkline_values(sparkline_data, target_points=KPI_TREND_POINTS, floor=0.0),
                "sql": value_res.get("executed_sql") or "",
                "trend_sql": trend_sql_out,
                "manual_config": item.get("manual_config") if isinstance(item.get("manual_config"), dict) else {},
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
                    "manual_config": _normalize_chart_manual_config_from_spec(
                        item.get("manual_config") if isinstance(item.get("manual_config"), dict) else {},
                        default_table=source_table_base,
                        default_chart_type=str(item.get("type") or "bar"),
                        kb_data=kb_data,
                    ),
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
                available_columns=column_names,
                logs=log,
                context=f"Filter Chart {item.get('chart_id', '')}",
            )
            c_data["id"] = item["chart_id"]
            c_data["sql"] = c_data.get("sql") or chart_res.get("executed_sql") or ""
            c_data["showDataLabels"] = item["showDataLabels"]
            c_data["manual_config"] = _normalize_chart_manual_config_from_spec(
                item.get("manual_config") if isinstance(item.get("manual_config"), dict) else {},
                default_table=source_table_base,
                default_chart_type=str(c_data.get("type") or item.get("type") or "bar"),
                kb_data=kb_data,
            )
            output["charts"].append(c_data)

        map_chart_refreshed = _build_map_chart_from_databricks(
            connection=connection,
            source_table_base=source_table_base,
            source_table_query=source_table_query,
            where_sql=where_sql,
            column_names=column_names,
            schema_columns=schema_columns,
            num_cols=num_cols,
            metric_profiles=_build_kb_metric_profiles(kb_data, schema_columns) if isinstance(kb_data, dict) else [],
            metric_name_hint=map_metric_name_hint,
            metric_formula_hint=map_metric_formula_hint,
            kb_data=kb_data,
            schema_context=schema_context,
            logs=log,
        )
        output["charts"] = [
            c for c in output.get("charts", [])
            if str((c or {}).get("type", "")).lower() != "india_map"
        ]
        output["charts"].insert(0, map_chart_refreshed)
        for i, c in enumerate(output["charts"]):
            c["id"] = f"chart_{i}"
        output["charts"] = output["charts"][:TOTAL_CHART_TARGET]

        output["__cache"] = {
            "filters": output.get("filters", []),
            "source_table": source_table_base,
            "date_range": output.get("date_range"),
            "selected_date_column": output.get("selected_date_column"),
        }
        return output
    finally:
        connection.close()

def generate_custom_chart_from_prompt_databricks(
    user_prompt,
    active_filters_json='{}',
    clarification_choice=None,
    allow_ambiguity_fallback=False,
    kb_module_name=None,
):
    connection = get_databricks_connection()
    llm_logs = _LiveLogBuffer(context="Custom Chart")
    try:
        include_sample_rows = _llm_include_sample_rows()
        if not include_sample_rows:
            llm_logs.append("[SECURITY] Databricks metadata-only mode enabled: LLM prompt excludes sample row values")

        kb_data = None
        resolved_kb_module = _resolve_kb_module_name(kb_module_name)
        if _is_kb_enabled():
            llm_logs.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=llm_logs)

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=llm_logs,
            kb_data=kb_data,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        metric_profiles = _build_kb_metric_profiles(kb_data, schema_columns)
        if metric_profiles:
            llm_logs.append(f"[KB-METRIC] Custom chart planner constrained to KB Section 4 metrics ({len(metric_profiles)})")
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        if _is_kb_enabled():
            if kb_data is not None:
                # OLD (Option 1): selective KB injection by intent.
                # kb_context = _build_targeted_schema_context_from_knowledge_base(
                #     kb_data,
                #     intent_text=f"custom chart request {user_prompt}",
                #     logs=llm_logs,
                # )
                # if not kb_context:
                #     kb_context = _build_schema_context_from_knowledge_base(kb_data)

                # NEW: full KB context injection
                kb_context = _build_schema_context_from_knowledge_base(kb_data)
                if kb_context:
                    llm_logs.append("[KB] Using knowledge base context from pgadmin_module")
                    effective_schema_context = kb_context
                else:
                    llm_logs.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                    effective_schema_context = schema_context
            else:
                llm_logs.append("[KB] Knowledge base unavailable, falling back to Databricks metadata")
                effective_schema_context = schema_context
        else:
            llm_logs.append("[KB] Disabled via KB_ENABLED=false, falling back to Databricks metadata")
            effective_schema_context = schema_context

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
        detected_metric = None
        relationships_dict = {}
        if isinstance(kb_data, dict):
            metrics_dict = _extract_metrics_from_kb(kb_data)
            relationships_dict = _extract_relationships_from_kb(kb_data)
            detected_metric = _detect_requested_metric(prompt_for_chart, metrics_dict)
            if isinstance(detected_metric, dict):
                llm_logs.append(f"[KB METRIC] Detected metric: {str(detected_metric.get('name') or '').strip()}")

        ai_plan, tokens_used = generate_custom_chart_plan(
            # OLD: using generic Databricks metadata
            # schema_context,
            effective_schema_context,
            prompt_for_chart,
            debug_logs=llm_logs,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
            metric_profiles=metric_profiles,
            detected_metric=detected_metric,
            relationships_dict=relationships_dict,
        )
        llm_logs.append(
            "[LLM RESPONSE] Custom Chart Plan Parse | "
            + json.dumps(
                {
                    "llm_called": True,
                    "has_plan": bool(ai_plan),
                    "has_sql": bool(str((ai_plan or {}).get("sql") or "").strip()) if isinstance(ai_plan, dict) else False,
                    "chart_type": str((ai_plan or {}).get("type") or "") if isinstance(ai_plan, dict) else "",
                    "title": str((ai_plan or {}).get("title") or "") if isinstance(ai_plan, dict) else "",
                    "tokens_used": tokens_used,
                },
                ensure_ascii=False,
            )
        )
        if not ai_plan:
            ai_plan = _default_custom_chart_plan_from_columns(
            schema_columns,
            user_prompt,
            table_name=DATABRICKS_LOGICAL_VIEW_NAME,
        )

        if isinstance(ai_plan, dict):
            sql_before_guard = str(ai_plan.get("sql") or "").strip()
            if sql_before_guard and not _validate_metric_formula_in_sql(sql_before_guard, detected_metric):
                fallback_candidates = [
                    str(ai_plan.get("x_column") or "").strip(),
                    str(date_column or "").strip(),
                    str(_find_col_case_insensitive(column_names, [str(ai_plan.get("xlabel") or "").strip().replace(" ", "_")]) or "").strip(),
                    str(date_cols[0] if date_cols else "").strip(),
                    str(column_names[0] if column_names else "").strip(),
                ]
                fallback_dimension = next((c for c in fallback_candidates if c), "")
                fallback_sql = _build_sql_from_metric_formula(detected_metric, fallback_dimension)
                if fallback_sql:
                    ai_plan["sql"] = fallback_sql
            ai_plan["sql"] = _validate_joins_in_sql(str(ai_plan.get("sql") or ""), relationships_dict)

        ai_plan = _enforce_requested_time_grain_on_chart_plan(ai_plan, prompt_for_chart, preferred_date_col=date_column)

        sql = str(ai_plan.get("sql", "")).replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
        sql, removed_ts_limit = _remove_limit_for_time_series_sql(sql, preferred_date_col=date_column)
        if removed_ts_limit:
            llm_logs.append("[TIME] Removed LIMIT from time-series chart SQL to include full selected period")
        sql, chart_contract_notes = _enforce_analysis_view_sql_contract(
            sql,
            available_columns=column_names,
        )
        for note in chart_contract_notes:
            llm_logs.append(f"[KB-GUARD] Custom Chart: {note}")
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
            sql, removed_ts_limit = _remove_limit_for_time_series_sql(sql, preferred_date_col=date_column)
            if removed_ts_limit:
                llm_logs.append("[TIME] Removed LIMIT from fallback time-series chart SQL to include full selected period")
            sql, chart_contract_notes = _enforce_analysis_view_sql_contract(
                sql,
                available_columns=column_names,
            )
            for note in chart_contract_notes:
                llm_logs.append(f"[KB-GUARD] Custom Chart (fallback): {note}")
            sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(sql)
            for note in guardrail_notes:
                llm_logs.append(f"[SECURITY] {note}")

        ai_plan["sql"] = sql
        ai_plan = _enrich_custom_chart_plan_metric_metadata(
            ai_plan,
            metric_profiles=metric_profiles,
            schema_columns=schema_columns,
            logs=llm_logs,
        )

        df, executed_sql = _execute_databricks_user_sql(
            connection,
            sql,
            source_table_base,
            query_source=source_table_query,
            where_sql=where_sql,
            available_columns=column_names,
            logs=llm_logs,
            context="Custom Chart",
        )

        if df is not None and df.empty:
            relaxed_sql = _relax_string_equality_predicates(sql)
            if relaxed_sql != sql:
                llm_logs.append("[FALLBACK] Custom Chart returned no rows; retrying with case-insensitive string matching")
                retry_df, retry_executed_sql = _execute_databricks_user_sql(
                    connection,
                    relaxed_sql,
                    source_table_base,
                    query_source=source_table_query,
                    where_sql=where_sql,
                    available_columns=column_names,
                    logs=llm_logs,
                    context="Custom Chart Fallback",
                )
                if retry_df is not None and not retry_df.empty:
                    df = retry_df
                    executed_sql = retry_executed_sql
                    llm_logs.append("[FALLBACK] Custom Chart fallback succeeded")

        chart_payload = _build_custom_chart_payload(ai_plan, df)
        chart_payload["sql"] = executed_sql
        chart_manual = _normalize_chart_manual_config_from_spec(
            ai_plan,
            default_table=source_table_base,
            default_chart_type=str(chart_payload.get("type") or ai_plan.get("type") or "bar"),
            kb_data=kb_data,
        )
        chart_payload["manual_config"] = chart_manual
        if str(chart_manual.get("metric_name") or "").strip():
            chart_payload["metric_name"] = str(chart_manual.get("metric_name") or "").strip()
        if str(chart_manual.get("metric_formula") or "").strip():
            chart_payload["metric_formula"] = str(chart_manual.get("metric_formula") or "").strip()
        return {
            "chart": chart_payload,
            "generated_sql": executed_sql,
            "tokens_used": tokens_used,
            "logs": llm_logs,
            "data_mode": "databricks",
        }
    finally:
        connection.close()


def _build_manual_config_table_catalog(kb_data):
    if not isinstance(kb_data, dict):
        return {"tables": [], "relationships": [], "metrics": []}

    selected_columns_map = _normalize_selected_columns_map(kb_data.get("selected_columns"))
    column_meta_map = _extract_knowledge_graph_column_meta(kb_data.get("knowledge_graph_data"))
    table_names = _extract_table_names(kb_data.get("tables"))

    ordered_tables = []
    seen_tables = set()

    def _ensure_table(table_name):
        name = str(table_name or "").strip()
        if not name:
            return None
        key = name.lower()
        if key in seen_tables:
            return next((t for t in ordered_tables if str(t.get("name", "")).lower() == key), None)
        entry = {"name": name, "columns": []}
        seen_tables.add(key)
        ordered_tables.append(entry)
        return entry

    for _, payload in (selected_columns_map or {}).items():
        t_name = str((payload or {}).get("table_name") or "").strip()
        if t_name:
            _ensure_table(t_name)

    for t_name in table_names:
        _ensure_table(t_name)

    for table in ordered_tables:
        t_name = str(table.get("name") or "").strip()
        t_key = t_name.lower()
        payload = selected_columns_map.get(t_key, {}) if isinstance(selected_columns_map, dict) else {}
        table_meta = column_meta_map.get(t_key, {}) if isinstance(column_meta_map, dict) else {}

        columns = []
        seen_cols = set()

        for col in (payload.get("columns") or []):
            col_name = str((col or {}).get("name") or "").strip()
            if not col_name:
                continue
            c_key = col_name.lower()
            if c_key in seen_cols:
                continue
            seen_cols.add(c_key)
            meta = table_meta.get(c_key, {}) if isinstance(table_meta, dict) else {}
            columns.append(
                {
                    "name": col_name,
                    "datatype": str((col or {}).get("datatype") or meta.get("datatype") or "").strip(),
                    "description": str((col or {}).get("description") or meta.get("description") or "").strip(),
                }
            )

        if not columns and isinstance(table_meta, dict):
            for col_name, meta in table_meta.items():
                resolved_name = str(col_name or "").strip()
                if not resolved_name:
                    continue
                c_key = resolved_name.lower()
                if c_key in seen_cols:
                    continue
                seen_cols.add(c_key)
                columns.append(
                    {
                        "name": resolved_name,
                        "datatype": str((meta or {}).get("datatype") or "").strip(),
                        "description": str((meta or {}).get("description") or "").strip(),
                    }
                )

        table["columns"] = columns

    return {
        "tables": ordered_tables,
        "relationships": _extract_relationship_lines(kb_data.get("relationships")),
        "metrics": _extract_metric_rows(kb_data.get("metrics_data")),
    }


def _resolve_manual_config_column(selected_column, available_columns):
    raw = str(selected_column or "").strip()
    if not raw:
        return ""
    cols = [str(c).strip() for c in (available_columns or []) if str(c).strip()]
    if not cols:
        return raw

    lookup = {c.lower(): c for c in cols}
    direct = lookup.get(raw.lower())
    if direct:
        return direct

    tail = raw.split(".")[-1].strip().lower()
    if tail in lookup:
        return lookup[tail]

    for c in cols:
        cl = c.lower()
        if cl.endswith(f"_{tail}") or cl.endswith(f".{tail}"):
            return c
    return tail or raw


def _manual_config_default_sql(x_col, y_col, agg, view_name=DATABRICKS_LOGICAL_VIEW_NAME):
    agg_norm = str(agg or "").strip().upper()
    if agg_norm not in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
        agg_norm = "COUNT"

    x_expr = _quote_identifier(x_col) if x_col else "'All'"
    y_identifier = _quote_identifier(y_col) if y_col else ""

    if agg_norm == "COUNT":
        y_expr = "COUNT(*)"
    elif y_identifier:
        y_expr = f"{agg_norm}({y_identifier})"
    else:
        y_expr = "COUNT(*)"

    return (
        f"SELECT {x_expr} AS x, {y_expr} AS y "
        f"FROM {view_name} "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 30"
    )


def _looks_like_geography_column(column_name):
    col = str(column_name or "").strip().lower()
    if not col:
        return False
    geo_tokens = [
        "region", "zone", "state", "territory", "city", "district", "area",
        "som", "asm", "geo",
    ]
    return any(tok in col for tok in geo_tokens)


def _normalize_column_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _is_map_region_dimension_column(column_name):
    raw = str(column_name or "").strip().lower()
    if not raw:
        return False
    norm = _normalize_column_token(raw)
    if not norm:
        return False
    if norm == "customerregionname":
        return True
    if norm == "regionname" or norm.endswith("regionname"):
        return True
    if "region" in norm:
        return True
    if "zone" in norm or "state" in norm or "territory" in norm:
        return True
    return False


def _find_map_region_dimension_column(available_columns, preferred_column=""):
    cols = [str(c).strip() for c in (available_columns or []) if str(c).strip()]
    if not cols:
        return ""

    preferred = str(preferred_column or "").strip()
    if preferred:
        preferred_resolved = _resolve_manual_config_column(preferred, cols)
        if _is_map_region_dimension_column(preferred_resolved):
            return preferred_resolved

    ranked = []
    for col in cols:
        if not _is_map_region_dimension_column(col):
            continue
        norm = _normalize_column_token(col)
        score = 0
        if norm == "customerregionname":
            score += 100
        elif norm == "regionname":
            score += 90
        elif "regionname" in norm:
            score += 80
        elif "region" in norm:
            score += 70
        elif "zone" in norm:
            score += 60
        elif "state" in norm or "territory" in norm:
            score += 50
        score += max(0, 20 - len(col))
        ranked.append((score, col.lower(), col))

    if not ranked:
        return ""
    ranked.sort(reverse=True)
    return ranked[0][2]


def _resolve_metric_expression_for_manual_map(
    metric_formula="",
    selected_metric_profile=None,
    y_column="",
    agg_raw="AUTO",
    column_type_lookup=None,
    available_columns=None,
):
    formula = str(metric_formula or "").strip()
    profile = selected_metric_profile if isinstance(selected_metric_profile, dict) else {}
    cols = [str(c).strip() for c in (available_columns or []) if str(c).strip()]
    ctype_lookup = column_type_lookup if isinstance(column_type_lookup, dict) else {}

    expr = ""
    if formula:
        if _metric_formula_looks_like_expression(formula):
            expr = formula
        elif formula.lower().startswith(("select ", "with ")):
            expr = _extract_primary_metric_expression_from_sql(formula)

    if not expr:
        expr = str(profile.get("expression") or "").strip()
    if not expr:
        expr = str(formula or "").strip() if _metric_formula_looks_like_expression(formula) else ""

    if expr:
        expr = _normalize_metric_expression_for_analysis_view(expr, cols)
        if expr:
            return expr

    y_col = str(y_column or "").strip()
    agg = str(agg_raw or "AUTO").strip().upper()
    if agg not in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
        y_type = str((ctype_lookup or {}).get(y_col.lower()) or "").lower()
        agg = "SUM" if (_is_numeric_dtype(y_type) and y_col) else "COUNT"

    if agg == "COUNT":
        return "COUNT(*)"
    if y_col:
        return f"{agg}({_quote_identifier(y_col)})"
    return "COUNT(*)"


def _build_manual_india_map_chart_payload(
    connection,
    source_table_base,
    source_table_query,
    where_sql,
    column_names,
    schema_columns,
    x_column,
    y_column,
    metric_name,
    metric_formula,
    selected_metric_profile=None,
    agg_raw="AUTO",
    x_column_raw="",
    y_column_raw="",
    x_table="",
    y_table="",
    logs=None,
):
    # Build baseline map once so we retain current full-map metadata behavior.
    base_num_cols = [c for c, t in (schema_columns or []) if _is_numeric_dtype(t)]
    baseline_map = _build_map_chart_from_databricks(
        connection=connection,
        source_table_base=source_table_base,
        source_table_query=source_table_query,
        where_sql=where_sql,
        column_names=column_names,
        schema_columns=schema_columns,
        num_cols=base_num_cols,
        use_llm_metric_selection=False,
        logs=logs,
    )

    region_col = str(x_column or "").strip()
    if not _looks_like_geography_column(region_col):
        detected_region, _ = _detect_region_column(schema_columns, column_names)
        if detected_region:
            if logs is not None:
                logs.append(
                    f"[MAP] Selected X column '{region_col or '(empty)'}' is not geographic; "
                    f"falling back to detected region column '{detected_region}'"
                )
            region_col = detected_region

    if not region_col:
        if logs is not None:
            logs.append("[MAP] No valid geography dimension selected/detected; falling back to baseline map")
        return baseline_map
    dimension_col = region_col

    col_type_lookup = {str(c).lower(): str(t) for c, t in (schema_columns or [])}
    metric_expr = _resolve_metric_expression_for_manual_map(
        metric_formula=metric_formula,
        selected_metric_profile=selected_metric_profile,
        y_column=y_column,
        agg_raw=agg_raw,
        column_type_lookup=col_type_lookup,
        available_columns=column_names,
    )
    region_ident = _quote_identifier(region_col)

    map_sql = (
        f"SELECT CAST({region_ident} AS STRING) AS x, {metric_expr} AS y "
        f"FROM {DATABRICKS_LOGICAL_VIEW_NAME} "
        f"WHERE {region_ident} IS NOT NULL "
        f"AND TRIM(CAST({region_ident} AS STRING)) != '' "
        "GROUP BY 1 ORDER BY 2 DESC"
    )

    base_col_names = {str(c).strip().lower() for c, _ in (_describe_databricks_table_columns(connection, source_table_base) or [])}
    expr_cols = _extract_metric_formula_columns(metric_expr, column_names)
    needed_cols = {str(region_col).strip().lower(), *(str(c).strip().lower() for c in expr_cols)}
    if dimension_col:
        needed_cols.add(str(dimension_col).strip().lower())
    needs_join_source = any(c and c not in base_col_names for c in needed_cols)
    map_query_source = source_table_query if needs_join_source else source_table_base

    try:
        map_df, executed_sql = _execute_databricks_user_sql(
            connection,
            map_sql,
            source_table_base,
            query_source=map_query_source,
            where_sql=where_sql,
            available_columns=column_names,
            logs=logs,
            context="Manual Configure India Map",
        )
    except Exception as e:
        if logs is not None:
            logs.append(f"[WARN] Manual map query failed: {str(e)}; using baseline map")
        return baseline_map

    if map_df is None or map_df.empty:
        if logs is not None:
            logs.append("[MAP] Manual map query returned no rows; using baseline map")
        return baseline_map

    map_df.columns = [str(c).lower() for c in map_df.columns]
    x_vals = map_df["x"].astype(str).tolist() if "x" in map_df else map_df.iloc[:, 0].astype(str).tolist()
    y_vals = map_df["y"].fillna(0).astype(float).tolist() if "y" in map_df else map_df.iloc[:, 1].fillna(0).astype(float).tolist()

    total = float(sum(abs(v) for v in y_vals))
    regions_meta = []
    for i in range(len(x_vals)):
        val = float(y_vals[i]) if i < len(y_vals) else 0.0
        pct = round((val / total * 100.0), 2) if total > 0 else 0.0
        regions_meta.append({
            "name": str(x_vals[i]),
            "value": val,
            "pct": pct,
        })

    base_meta = baseline_map.get("map_meta") if isinstance(baseline_map, dict) else {}
    region_to_states = base_meta.get("region_to_states") if isinstance(base_meta, dict) else {}
    mapping_type = "business_region" if _looks_like_geography_column(region_col) and any(
        tok in str(region_col).lower() for tok in ["region", "zone", "som", "asm", "area"]
    ) else "direct"

    metric_label = str(metric_name or "").strip() or str(
        (selected_metric_profile or {}).get("name")
        or (selected_metric_profile or {}).get("primary_column")
        or y_column
        or "Value"
    ).strip()

    return {
        "id": "custom_generated",
        "title": f"{metric_label} Map",
        "type": "india_map",
        "xlabel": str(region_col).replace("_", " ").title(),
        "ylabel": metric_label,
        "sql": executed_sql,
        "x": x_vals,
        "y": y_vals,
        "z": [],
        "region_col": region_col,
        "metric_col": metric_expr,
        "columns": [],
        "rows": [],
        "labels": [],
        "parents": [],
        "values": [],
        "measure": [],
        "map_meta": {
            "region_col": region_col,
            "dimension_col": dimension_col or region_col,
            "metric_col": metric_expr,
            "metric_label": metric_label,
            "total": total,
            "regions": regions_meta,
            "region_to_states": region_to_states if isinstance(region_to_states, dict) else {},
            "mapping_type": mapping_type,
        },
        "manual_config": {
            "x_table": x_table,
            "x_column": region_col,
            "y_table": y_table,
            "y_column": y_column_raw or y_column,
            "aggregation": agg_raw if agg_raw in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"} else "AUTO",
            "chart_type": "india_map",
            "metric_name": metric_name,
            "metric_formula": metric_formula,
            "metric_unit": str((selected_metric_profile or {}).get("unit_type") or "").strip().lower(),
            "metric_axis": "y",
            "dimension_axis": "x",
        },
    }


def _build_india_map_chart_payload_from_df(
    map_df,
    executed_sql,
    region_col,
    metric_label,
    metric_expr,
    x_table,
    y_table,
    y_column_raw,
    y_column,
    agg_raw,
    metric_name,
    metric_formula,
    selected_metric_profile=None,
    baseline_map=None,
):
    if map_df is None or map_df.empty:
        return None

    work = map_df.copy()
    work.columns = [str(c).lower() for c in work.columns]
    x_vals = work["x"].astype(str).tolist() if "x" in work else work.iloc[:, 0].astype(str).tolist()
    y_vals = work["y"].fillna(0).astype(float).tolist() if "y" in work else work.iloc[:, 1].fillna(0).astype(float).tolist()

    if not x_vals or not y_vals:
        return None

    total = float(sum(abs(v) for v in y_vals))
    regions_meta = []
    for i in range(len(x_vals)):
        val = float(y_vals[i]) if i < len(y_vals) else 0.0
        pct = round((val / total * 100.0), 2) if total > 0 else 0.0
        regions_meta.append(
            {
                "name": str(x_vals[i]),
                "value": val,
                "pct": pct,
            }
        )

    base_meta = baseline_map.get("map_meta") if isinstance(baseline_map, dict) else {}
    region_to_states = base_meta.get("region_to_states") if isinstance(base_meta, dict) else {}
    mapping_type = "business_region" if _looks_like_geography_column(region_col) and any(
        tok in str(region_col).lower() for tok in ["region", "zone", "som", "asm", "area"]
    ) else "direct"

    return {
        "id": "custom_generated",
        "title": f"{metric_label} Map",
        "type": "india_map",
        "xlabel": str(region_col).replace("_", " ").title(),
        "ylabel": metric_label,
        "sql": executed_sql,
        "x": x_vals,
        "y": y_vals,
        "z": [],
        "region_col": region_col,
        "metric_col": metric_expr,
        "columns": [],
        "rows": [],
        "labels": [],
        "parents": [],
        "values": [],
        "measure": [],
        "map_meta": {
            "region_col": region_col,
            "dimension_col": region_col,
            "metric_col": metric_expr,
            "metric_label": metric_label,
            "total": total,
            "regions": regions_meta,
            "region_to_states": region_to_states if isinstance(region_to_states, dict) else {},
            "mapping_type": mapping_type,
        },
        "manual_config": {
            "x_table": x_table,
            "x_column": region_col,
            "y_table": y_table,
            "y_column": y_column_raw or y_column,
            "aggregation": agg_raw if agg_raw in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"} else "AUTO",
            "chart_type": "india_map",
            "metric_name": metric_name,
            "metric_formula": metric_formula,
            "metric_unit": str((selected_metric_profile or {}).get("unit_type") or "").strip().lower(),
            "metric_axis": "y",
            "dimension_axis": "x",
        },
    }


def generate_manual_configured_chart_databricks(
    config_payload,
    active_filters_json="{}",
    kb_module_name=None,
):
    payload = config_payload if isinstance(config_payload, dict) else {}
    x_table = str(payload.get("x_table") or "").strip()
    x_column_raw = str(payload.get("x_column") or "").strip()
    y_table = str(payload.get("y_table") or "").strip()
    y_column_raw = str(payload.get("y_column") or "").strip()
    agg_raw = str(payload.get("aggregation") or "AUTO").strip().upper()
    chart_type = str(payload.get("chart_type") or "bar").strip().lower()
    metric_name = str(payload.get("metric_name") or "").strip()
    metric_formula = str(payload.get("metric_formula") or "").strip()
    metric_axis_hint = str(payload.get("metric_axis") or "").strip().lower()

    allowed_manual_types = set(ALLOWED_CUSTOM_CHART_TYPES) | {"india_map"}
    if chart_type not in allowed_manual_types:
        chart_type = "bar"

    connection = get_databricks_connection()
    llm_logs = _LiveLogBuffer(context="Configure Chart")
    try:
        llm_logs.append(
            "[LLM REQUEST] Configure Chart Input | "
            + json.dumps(
                {
                    "chart_type": chart_type,
                    "x_table": x_table,
                    "x_column": x_column_raw,
                    "y_table": y_table,
                    "y_column": y_column_raw,
                    "aggregation": agg_raw,
                    "metric_name": metric_name,
                    "metric_axis": metric_axis_hint or "y",
                    "filters_chars": len(str(active_filters_json or "")),
                },
                ensure_ascii=False,
            )
        )
        include_sample_rows = _llm_include_sample_rows()
        kb_data = None
        resolved_kb_module = _resolve_kb_module_name(kb_module_name)
        if _is_kb_enabled():
            llm_logs.append(f"[KB] Active module: {resolved_kb_module}")
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=llm_logs)

        source_model = _build_databricks_virtual_source(
            connection,
            include_sample_rows=include_sample_rows,
            logs=llm_logs,
            kb_data=kb_data,
        )
        source_table_base = source_model["base_table"]
        source_table_query = source_model["query_source"]
        schema_context = source_model["schema_context"]
        schema_columns = source_model["schema_columns"]
        column_names = [c for c, _ in schema_columns]
        column_type_lookup = {str(c).lower(): str(t) for c, t in schema_columns}
        date_cols = [c for c, t in schema_columns if _is_date_dtype(t)]
        date_column = _resolve_selected_date_column(active_filters_json, date_cols)

        x_column = _resolve_manual_config_column(x_column_raw, column_names)
        y_column = _resolve_manual_config_column(y_column_raw, column_names)
        metric_profiles = _build_kb_metric_profiles(kb_data, schema_columns)
        selected_metric_profile = _resolve_metric_profile_for_hints(
            metric_profiles,
            metric_name=metric_name,
            metric_formula=metric_formula,
            preferred_columns=[y_column_raw or y_column, x_column_raw or x_column],
        )
        if selected_metric_profile:
            metric_name = metric_name or str(selected_metric_profile.get("name") or "").strip()
            metric_formula = metric_formula or str(selected_metric_profile.get("formula") or "").strip()

        if chart_type == "india_map":
            if not selected_metric_profile:
                raise ValueError("For India Map, select one metric from KB Section 4 only.")
            metric_name = str(selected_metric_profile.get("name") or "").strip()
            metric_formula = str(selected_metric_profile.get("formula") or "").strip()
            if not metric_name or not metric_formula:
                raise ValueError("Selected KB metric is incomplete for India Map.")
            resolved_dim_col = _find_map_region_dimension_column(
                column_names,
                preferred_column=(x_column_raw or x_column),
            )
            if not resolved_dim_col:
                raise ValueError("For India Map, dimension must be customer_region_name (region).")
            x_column = resolved_dim_col
            x_column_raw = resolved_dim_col

        resolved_x_table = _resolve_manual_config_table_for_column(
            x_column_raw or x_column,
            kb_data=kb_data,
            default_table=source_table_base,
        )
        resolved_y_table = _resolve_manual_config_table_for_column(
            y_column_raw or y_column,
            kb_data=kb_data,
            default_table=source_table_base,
        )

        where_sql, filter_count = _build_databricks_where_clause(
            active_filters_json,
            column_names,
            date_column=date_column,
            column_type_lookup=column_type_lookup,
        )
        if filter_count > 0:
            llm_logs.append(f"[FILTER] Applied {filter_count} filter(s) in manual chart configure")

        map_mode = chart_type == "india_map"

        def _deterministic_map_fallback(reason_text, tokens_used_for_return=0):
            llm_logs.append(f"[FALLBACK] {reason_text}; using deterministic India map SQL")
            fallback_chart = _build_manual_india_map_chart_payload(
                connection=connection,
                source_table_base=source_table_base,
                source_table_query=source_table_query,
                where_sql=where_sql,
                column_names=column_names,
                schema_columns=schema_columns,
                x_column=x_column,
                y_column=y_column,
                metric_name=metric_name,
                metric_formula=metric_formula,
                selected_metric_profile=selected_metric_profile,
                agg_raw=agg_raw,
                x_column_raw=x_column_raw,
                y_column_raw=y_column_raw,
                x_table=resolved_x_table or x_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                y_table=resolved_y_table or y_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                logs=llm_logs,
            )
            fallback_chart["type"] = "india_map"
            llm_logs.append(
                "[LLM RESPONSE] Configure SQL | "
                + json.dumps(
                    {
                        "mode": "deterministic_map_fallback",
                        "llm_called": False,
                        "selected_metric": metric_name,
                        "selected_dimension": x_column_raw or x_column,
                        "generated_chart_type": "india_map",
                        "generated_sql_chars": len(str(fallback_chart.get("sql") or "")),
                    },
                    ensure_ascii=False,
                )
            )
            return {
                "chart": fallback_chart,
                "generated_sql": str(fallback_chart.get("sql") or ""),
                "tokens_used": tokens_used_for_return,
                "logs": llm_logs,
                "data_mode": "databricks",
            }

        kb_context = _build_schema_context_from_knowledge_base(kb_data) if isinstance(kb_data, dict) else ""
        relationship_lines = _extract_relationship_lines((kb_data or {}).get("relationships"))
        relationship_text = "\n".join(relationship_lines[:80]) if relationship_lines else "No relationships available."

        if map_mode:
            system_prompt = (
                "You are a senior Databricks SQL analyst building one India map chart query for an AI dashboard.\n"
                "Return ONLY JSON with keys: sql, title, xlabel, ylabel, chart_type.\n"
                "chart_type must be india_map.\n"
                f"Use {DATABRICKS_LOGICAL_VIEW_NAME} as the logical source table in SQL.\n"
                "Alias output columns exactly as x and y.\n"
                "x must represent region names; y must represent the selected metric value.\n"
                "Respect KB Section 3 explicit relationships for join-path logic when mapping metric/dimension columns; do not invent unsupported columns or join keys.\n"
                "Use the selected metric formula semantics exactly (including joins/expressions needed), adapted to analysis_view columns."
            )
            system_prompt = f"{system_prompt.rstrip()}\n\n{_sql_generation_safety_rules_block(DATABRICKS_LOGICAL_VIEW_NAME)}"
            user_prompt = (
                "Manual India Map configuration request:\n"
                f"- Region dimension column (fixed): {x_column_raw or x_column or '(unspecified)'}\n"
                f"- Metric helper column: {y_column_raw or y_column or '(unspecified)'}\n"
                f"- Aggregation preference: {agg_raw or 'AUTO'}\n"
                f"- Selected metric name: {metric_name or '(none)'}\n"
                f"- Selected metric formula: {metric_formula or '(none)'}\n\n"
                f"Schema context:\n{schema_context}\n\n"
                f"Knowledge-base schema context:\n{kb_context or 'Not available'}\n\n"
                f"Relationships:\n{relationship_text}\n\n"
                "Generate SQL for region-wise India map output with x=region and y=metric value.\n"
                "Do not use generic COUNT(*) unless the selected metric genuinely requires it.\n\n"
                f"{_sql_generation_final_checklist_block(DATABRICKS_LOGICAL_VIEW_NAME)}"
            )
        else:
            system_prompt = (
                "You are a senior Databricks SQL analyst building one chart query for an AI dashboard.\n"
                "Return ONLY JSON with keys: sql, title, xlabel, ylabel, chart_type.\n"
                f"chart_type must be one of: {', '.join(sorted(ALLOWED_CUSTOM_CHART_TYPES))}.\n"
                f"Use {DATABRICKS_LOGICAL_VIEW_NAME} as the logical source table in SQL.\n"
                "Alias output columns exactly as x and y.\n"
                "Ensure SQL is valid Databricks SQL. Use LIMIT 30 only for category charts; do NOT limit date/time trend series.\n"
                "Respect KB Section 3 explicit relationships for join-path logic when mapping metric/dimension columns; do not invent unsupported columns or join keys."
            )
            system_prompt = f"{system_prompt.rstrip()}\n\n{_sql_generation_safety_rules_block(DATABRICKS_LOGICAL_VIEW_NAME)}"
            user_prompt = (
                f"Manual configuration request:\n"
                f"- X axis column: {x_column_raw or '(unspecified)'}\n"
                f"- Y axis column: {y_column_raw or '(unspecified)'}\n"
                f"- Aggregation: {agg_raw or 'AUTO'}\n"
                f"- Chart type: {chart_type}\n"
                f"- Selected metric name: {metric_name or '(none)'}\n"
                f"- Selected metric formula (temporary, editable): {metric_formula or '(none)'}\n\n"
                f"- Preferred metric axis: {metric_axis_hint or 'y'}\n\n"
                f"Schema context:\n{schema_context}\n\n"
                f"Knowledge-base schema context:\n{kb_context or 'Not available'}\n\n"
                f"Relationships:\n{relationship_text}\n\n"
                "If aggregation is AUTO, choose the best aggregation based on selected columns and chart semantics.\n"
                "Use the selected metric (name/formula) as the measure and pair it with one dimension/date axis.\n"
                "If a metric formula is provided, treat it as temporary chart-only guidance for this request and adapt it to valid analysis_view columns.\n\n"
                f"{_sql_generation_final_checklist_block(DATABRICKS_LOGICAL_VIEW_NAME)}"
            )

        llm_raw, tokens_used = call_ai_with_retry(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
            debug_logs=llm_logs,
            context="Configure SQL",
        )

        parsed = None
        try:
            parsed = json.loads(llm_raw) if llm_raw else None
        except Exception:
            parsed = None

        ai_sql = str((parsed or {}).get("sql") or "").strip()
        ai_title = str((parsed or {}).get("title") or "").strip()
        ai_xlabel = str((parsed or {}).get("xlabel") or "").strip()
        ai_ylabel = str((parsed or {}).get("ylabel") or "").strip()
        ai_chart_type = str((parsed or {}).get("chart_type") or chart_type).strip().lower()
        allowed_ai_types = set(ALLOWED_CUSTOM_CHART_TYPES) | {"india_map"}
        if ai_chart_type not in allowed_ai_types:
            ai_chart_type = chart_type
        if map_mode:
            ai_chart_type = "india_map"
        llm_logs.append(
            "[LLM RESPONSE] Configure SQL Parse | "
            + json.dumps(
                {
                    "llm_called": True,
                    "chart_type": ai_chart_type,
                    "has_sql": bool(ai_sql),
                    "title": ai_title,
                    "xlabel": ai_xlabel,
                    "ylabel": ai_ylabel,
                    "tokens_used": tokens_used,
                },
                ensure_ascii=False,
            )
        )

        if not ai_sql:
            if map_mode:
                return _deterministic_map_fallback("Configure SQL LLM output missing SQL", tokens_used_for_return=tokens_used)
            llm_logs.append("[FALLBACK] Configure SQL LLM output missing SQL; applying deterministic fallback query")
            fallback_agg = agg_raw
            if fallback_agg == "AUTO":
                y_type = str(column_type_lookup.get(str(y_column).lower()) or "").lower()
                fallback_agg = "SUM" if _is_numeric_dtype(y_type) else "COUNT"
            ai_sql = _manual_config_default_sql(x_column, y_column, fallback_agg, view_name=DATABRICKS_LOGICAL_VIEW_NAME)

        ai_sql = ai_sql.replace("master_view", DATABRICKS_LOGICAL_VIEW_NAME).replace("final_view", DATABRICKS_LOGICAL_VIEW_NAME).strip()
        ai_sql, removed_ts_limit = _remove_limit_for_time_series_sql(ai_sql, preferred_date_col=date_column)
        if removed_ts_limit:
            llm_logs.append("[TIME] Removed LIMIT from time-series configure SQL to include full selected period")

        relationships_dict = _extract_relationships_from_kb(kb_data) if isinstance(kb_data, dict) else {}
        if str(ai_sql).strip():
            metric_guard_payload = None
            if isinstance(selected_metric_profile, dict):
                metric_guard_payload = {
                    "name": str(selected_metric_profile.get("name") or metric_name or "").strip(),
                    "formula": str(selected_metric_profile.get("formula") or metric_formula or "").strip(),
                }
            else:
                metrics_dict = _extract_metrics_from_kb(kb_data) if isinstance(kb_data, dict) else {}
                metric_hint_text = " ".join(
                    [
                        str(metric_name or "").strip(),
                        str(metric_formula or "").strip(),
                        str(y_column_raw or "").strip(),
                        str(y_column or "").strip(),
                        str(ai_ylabel or "").strip(),
                        str(ai_title or "").strip(),
                    ]
                ).strip()
                detected_metric = _detect_requested_metric(metric_hint_text, metrics_dict)
                if isinstance(detected_metric, dict):
                    metric_guard_payload = {
                        "name": str(detected_metric.get("name") or metric_name or "").strip(),
                        "formula": str(detected_metric.get("formula") or metric_formula or "").strip(),
                    }
                elif str(metric_name or "").strip():
                    metric_guard_payload = {
                        "name": str(metric_name or "").strip(),
                        "formula": str(metric_formula or "").strip(),
                    }

            if isinstance(metric_guard_payload, dict):
                metric_guard_name_l = str(metric_guard_payload.get("name") or "").strip().lower()
                time_like_axis = any(
                    tok in str(x_column or x_column_raw or "").strip().lower()
                    for tok in ["date", "day", "week", "month", "year", "time"]
                )
                ulpo_time_dim = (
                    metric_guard_name_l == "ulpo"
                    and time_like_axis
                )
                if ulpo_time_dim:
                    ulpo_sql = _build_sql_from_metric_formula(
                        metric_guard_payload,
                        str(x_column or x_column_raw or "").strip(),
                    )
                    if ulpo_sql:
                        llm_logs.append(
                            "[METRIC GUARD] Manual Configure Chart aligned ULPO metric grain to time dimension."
                        )
                        ai_sql = ulpo_sql
                suspicious_eco_time_dim = metric_guard_name_l == "suspicious eco" and time_like_axis
                if suspicious_eco_time_dim:
                    suspicious_sql = _build_sql_from_metric_formula(
                        metric_guard_payload,
                        str(x_column or x_column_raw or "").strip(),
                    )
                    if suspicious_sql:
                        llm_logs.append(
                            "[METRIC GUARD] Manual Configure Chart aligned Suspicious ECO metric grain to month-level time dimension."
                        )
                        ai_sql = suspicious_sql
                if not _validate_metric_formula_in_sql(ai_sql, metric_guard_payload):
                    dimension_col = str(x_column or x_column_raw or "").strip()
                    rebuilt_sql = _build_sql_from_metric_formula(
                        metric_guard_payload,
                        dimension_col,
                        dimension_table=(resolved_x_table if resolved_x_table and resolved_x_table != source_table_base else None),
                    )
                    if rebuilt_sql:
                        llm_logs.append(
                            f"[METRIC GUARD] Manual Configure Chart rebuilt SQL from KB metric formula for "
                            f"{str(metric_guard_payload.get('name') or '').strip()}."
                        )
                        ai_sql = rebuilt_sql
        ai_sql = _validate_joins_in_sql(ai_sql, relationships_dict)

        ai_sql, chart_contract_notes = _enforce_analysis_view_sql_contract(
            ai_sql,
            available_columns=column_names,
        )
        for note in chart_contract_notes:
            llm_logs.append(f"[KB-GUARD] Manual Configure: {note}")
        ai_sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(ai_sql)
        for note in guardrail_notes:
            llm_logs.append(f"[SECURITY] {note}")

        if not _safe_custom_sql(ai_sql):
            if map_mode:
                return _deterministic_map_fallback("Manual Configure SQL failed safety checks", tokens_used_for_return=tokens_used)
            llm_logs.append("[FALLBACK] Manual Configure SQL failed safety checks; switching to deterministic fallback query")
            fallback_agg = agg_raw
            if fallback_agg == "AUTO":
                y_type = str(column_type_lookup.get(str(y_column).lower()) or "").lower()
                fallback_agg = "SUM" if _is_numeric_dtype(y_type) else "COUNT"
            ai_sql = _manual_config_default_sql(x_column, y_column, fallback_agg, view_name=DATABRICKS_LOGICAL_VIEW_NAME)
            ai_sql, chart_contract_notes = _enforce_analysis_view_sql_contract(
                ai_sql,
                available_columns=column_names,
            )
            for note in chart_contract_notes:
                llm_logs.append(f"[KB-GUARD] Manual Configure (fallback): {note}")
            ai_sql, guardrail_notes = _apply_sql_security_and_cost_guardrails(ai_sql)
            for note in guardrail_notes:
                llm_logs.append(f"[SECURITY] {note}")

        try:
            df, executed_sql = _execute_databricks_user_sql(
                connection,
                ai_sql,
                source_table_base,
                query_source=source_table_query,
                where_sql=where_sql,
                available_columns=column_names,
                logs=llm_logs,
                context="Manual Configure Chart",
            )
        except Exception as e:
            if map_mode:
                return _deterministic_map_fallback(
                    f"Manual Configure map SQL execution failed: {str(e)}",
                    tokens_used_for_return=tokens_used,
                )
            raise

        if map_mode:
            baseline_map = _build_map_chart_from_databricks(
                connection=connection,
                source_table_base=source_table_base,
                source_table_query=source_table_query,
                where_sql=where_sql,
                column_names=column_names,
                schema_columns=schema_columns,
                num_cols=[c for c, t in (schema_columns or []) if _is_numeric_dtype(t)],
                metric_profiles=metric_profiles,
                metric_name_hint=metric_name,
                metric_formula_hint=metric_formula,
                kb_data=kb_data,
                schema_context=schema_context,
                use_llm_metric_selection=False,
                logs=llm_logs,
            )
            metric_label = str(metric_name or "").strip() or str(
                (selected_metric_profile or {}).get("name")
                or (selected_metric_profile or {}).get("primary_column")
                or y_column
                or "Value"
            ).strip()
            metric_expr = _resolve_metric_expression_for_manual_map(
                metric_formula=metric_formula,
                selected_metric_profile=selected_metric_profile,
                y_column=y_column,
                agg_raw=agg_raw,
                column_type_lookup=column_type_lookup,
                available_columns=column_names,
            )
            chart_payload = _build_india_map_chart_payload_from_df(
                map_df=df,
                executed_sql=executed_sql,
                region_col=x_column or x_column_raw,
                metric_label=metric_label,
                metric_expr=metric_expr,
                x_table=resolved_x_table or x_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                y_table=resolved_y_table or y_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
                y_column_raw=y_column_raw,
                y_column=y_column,
                agg_raw=agg_raw,
                metric_name=metric_name,
                metric_formula=metric_formula,
                selected_metric_profile=selected_metric_profile,
                baseline_map=baseline_map,
            )
            if not chart_payload:
                return _deterministic_map_fallback("LLM map SQL returned no rows", tokens_used_for_return=tokens_used)
            chart_payload["type"] = "india_map"
            if ai_title:
                chart_payload["title"] = ai_title
            if ai_xlabel:
                chart_payload["xlabel"] = ai_xlabel
            if ai_ylabel:
                chart_payload["ylabel"] = ai_ylabel
            llm_logs.append(
                "[LLM RESPONSE] Configure SQL | "
                + json.dumps(
                    {
                        "mode": "llm_map",
                        "llm_called": True,
                        "selected_metric": metric_name,
                        "selected_dimension": x_column_raw or x_column,
                        "generated_chart_type": "india_map",
                        "generated_sql_chars": len(str(executed_sql or "")),
                    },
                    ensure_ascii=False,
                )
            )
            return {
                "chart": chart_payload,
                "generated_sql": executed_sql,
                "tokens_used": tokens_used,
                "logs": llm_logs,
                "data_mode": "databricks",
            }

        plan = {
            "type": ai_chart_type,
            "xlabel": ai_xlabel or x_column_raw or x_column or "X Axis",
            "ylabel": ai_ylabel or metric_name or y_column_raw or y_column or "Y Axis",
            "title": ai_title or f"{(metric_name or y_column_raw or y_column or 'Metric')} by {(x_column_raw or x_column or 'Dimension')}",
            "sql": executed_sql,
        }
        chart_payload = _build_custom_chart_payload(plan, df)
        chart_payload["sql"] = executed_sql
        chart_payload["type"] = ai_chart_type
        metric_axis = "x" if metric_axis_hint == "x" else "y"
        if selected_metric_profile:
            x_match = str(x_column_raw or x_column or "").strip().lower() in [str(c).strip().lower() for c in (selected_metric_profile.get("columns") or [])]
            y_match = str(y_column_raw or y_column or "").strip().lower() in [str(c).strip().lower() for c in (selected_metric_profile.get("columns") or [])]
            if x_match and not y_match:
                metric_axis = "x"
        chart_payload["manual_config"] = {
            "x_table": resolved_x_table or x_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
            "x_column": x_column_raw or x_column,
            "y_table": resolved_y_table or y_table or _extract_table_tail(source_table_base) or str(source_table_base or "analysis_view"),
            "y_column": y_column_raw or y_column,
            "aggregation": agg_raw if agg_raw in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"} else "AUTO",
            "chart_type": ai_chart_type,
            "metric_name": metric_name,
            "metric_formula": metric_formula,
            "metric_unit": str((selected_metric_profile or {}).get("unit_type") or "").strip().lower(),
            "metric_axis": metric_axis,
            "dimension_axis": "x" if metric_axis == "y" else "y",
        }

        return {
            "chart": chart_payload,
            "generated_sql": executed_sql,
            "tokens_used": tokens_used,
            "logs": llm_logs,
            "data_mode": "databricks",
        }
    finally:
        connection.close()


def generate_manual_configured_kpi_databricks(
    config_payload,
    active_filters_json="{}",
    kb_module_name=None,
):
    payload = config_payload if isinstance(config_payload, dict) else {}
    table_name = str(payload.get("table_name") or payload.get("x_table") or "").strip()
    column_name = str(payload.get("column_name") or payload.get("y_column") or payload.get("x_column") or "").strip()
    aggregation = str(payload.get("aggregation") or "AUTO").strip().upper()
    metric_name = str(payload.get("metric_name") or "").strip()
    metric_formula = str(payload.get("metric_formula") or "").strip()
    logs = []
    logs.append(
        "[LLM REQUEST] Configure KPI Input | "
        + json.dumps(
            {
                "table_name": table_name,
                "column_name": column_name,
                "aggregation": aggregation,
                "metric_name": metric_name,
                "filters_chars": len(str(active_filters_json or "")),
            },
            ensure_ascii=False,
        )
    )

    resolved_kb_module = _resolve_kb_module_name(kb_module_name)
    kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=logs)
    metric_profiles = _build_kb_metric_profiles(kb_data, [])
    selected_metric_profile = _resolve_metric_profile_for_hints(
        metric_profiles,
        metric_name=metric_name,
        metric_formula=metric_formula,
        preferred_columns=[column_name],
    )
    if selected_metric_profile:
        metric_name = metric_name or str(selected_metric_profile.get("name") or "").strip()
        metric_formula = metric_formula or str(selected_metric_profile.get("formula") or "").strip()
    resolved_table = _resolve_manual_config_table_for_column(
        column_name,
        kb_data=kb_data,
        default_table=(table_name or "analysis_view"),
    )

    prompt_lines = [
        "Configure one KPI based on manual selection.",
        f"Column: {column_name or '(unspecified)'}",
        f"Aggregation: {aggregation or 'AUTO'}",
        f"Metric Name: {metric_name or '(none)'}",
        f"Metric Formula (temporary, editable): {metric_formula or '(none)'}",
        "Use selected metric as the KPI measure and keep output metric-driven.",
        "Follow KB Section 3 explicit relationships for metric/dimension mapping and do not invent join keys.",
        "Use active dashboard filters exactly.",
        "Do not persist or modify metric formulas in KB/database.",
        "Return a KPI with value and trend SQL suitable for dashboard KPI card.",
    ]
    prompt_text = "\n".join(prompt_lines)

    result = generate_custom_kpi_from_prompt_databricks(
        prompt_text,
        active_filters_json=active_filters_json,
        allow_ambiguity_fallback=True,
        kb_module_name=kb_module_name,
    )

    if isinstance(result, dict) and isinstance(result.get("kpi"), dict):
        kpi_payload = result.get("kpi") or {}
        kpi_payload["manual_config"] = {
            "table_name": resolved_table or table_name,
            "column_name": column_name,
            "aggregation": aggregation if aggregation in {"SUM", "COUNT", "AVG", "MIN", "MAX", "AUTO"} else "AUTO",
            "metric_name": metric_name,
            "metric_formula": metric_formula,
            "metric_unit": str((selected_metric_profile or {}).get("unit_type") or "").strip().lower(),
        }
        result["kpi"] = kpi_payload
        logs.append(
            "[LLM RESPONSE] Configure KPI Output | "
            + json.dumps(
                {
                    "label": str(kpi_payload.get("label") or ""),
                    "has_sql": bool(kpi_payload.get("sql")),
                    "has_trend_sql": bool(kpi_payload.get("trend_sql")),
                },
                ensure_ascii=False,
            )
        )
    if isinstance(result, dict):
        existing_logs = result.get("logs")
        combined = []
        if isinstance(existing_logs, list):
            combined.extend(existing_logs)
        combined.extend(list(logs))
        result["logs"] = combined
    return result


def _wireframe_keyword_domain(prompt_text):
    text = str(prompt_text or "").lower()
    mapping = [
        (("invoice", "distributor", "sku", "fmcg", "retail", "sales"), "FMCG Sales Analytics"),
        (("patient", "hospital", "clinic", "doctor", "bed"), "Healthcare Operations"),
        (("student", "school", "course", "exam", "admission"), "Education Performance"),
        (("shipment", "warehouse", "delivery", "logistics", "fleet"), "Supply Chain & Logistics"),
        (("ticket", "resolution", "sla", "incident", "support"), "Customer Support Operations"),
        (("campaign", "funnel", "lead", "conversion", "marketing"), "Marketing Performance"),
        (("subscription", "churn", "mrr", "arr", "saas"), "SaaS Growth Analytics"),
        (("expense", "budget", "finance", "cost", "profit"), "Financial Operations"),
        (("factory", "production", "oee", "downtime", "yield"), "Manufacturing Performance"),
        (("employee", "attrition", "hiring", "headcount", "hr"), "People Analytics"),
    ]
    for keys, domain in mapping:
        if any(k in text for k in keys):
            return domain
    return "Business Performance"


def _wireframe_fallback_blueprint(prompt_text, kpi_count=6):
    domain = _wireframe_keyword_domain(prompt_text)
    defaults = {
        "FMCG Sales Analytics": [
            ("Total Gross Value", "currency", "up"),
            ("Total Net Amount", "currency", "up"),
            ("Invoice Quantity", "count", "up"),
            ("Unique Customers", "count", "up"),
            ("Average Order Value", "currency", "up"),
            ("Return Rate", "percent", "down"),
        ],
        "Healthcare Operations": [
            ("Patient Volume", "count", "up"),
            ("Average Wait Time", "duration", "down"),
            ("Bed Occupancy", "percent", "up"),
            ("Readmission Rate", "percent", "down"),
            ("Procedure Throughput", "count", "up"),
            ("Satisfaction Score", "score", "up"),
        ],
        "Business Performance": [
            ("Total Revenue", "currency", "up"),
            ("Total Cost", "currency", "down"),
            ("Net Margin", "percent", "up"),
            ("Active Customers", "count", "up"),
            ("Order Volume", "count", "up"),
            ("Growth Rate", "percent", "up"),
        ],
    }
    kpi_defs = defaults.get(domain, defaults["Business Performance"])[: max(3, min(8, int(kpi_count or 6)))]
    return {
        "domain": domain,
        "wireframe_title": f"{domain} KPI Wireframe",
        "assumptions": [
            "Wireframe uses synthetic sample data only.",
            "KPIs are inferred from the written requirement and optional reference image.",
            "Use this layout for stakeholder alignment before connecting live data.",
        ],
        "kpis": [
            {
                "label": label,
                "unit": unit,
                "direction": direction,
                "chart": "line",
                "intent": f"Track {label.lower()} trend over time.",
            }
            for (label, unit, direction) in kpi_defs
        ],
    }


def _parse_wireframe_llm_json(raw_text):
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    text = str(raw_text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _seed_from_text(text):
    digest = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _format_wireframe_value(value, unit):
    n = float(value)
    unit_s = str(unit or "").lower()
    if unit_s == "percent":
        return f"{n:.1f}%"
    if unit_s == "duration":
        if n >= 60:
            return f"{n / 60:.1f} hrs"
        return f"{n:.0f} min"
    if unit_s == "score":
        return f"{n:.2f}/5"
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def _wireframe_month_labels(periods):
    now = datetime.utcnow()
    labels = []
    year = now.year
    month = now.month
    for _ in range(periods):
        labels.append(datetime(year, month, 1).strftime("%b %Y"))
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    labels.reverse()
    return labels


def _generate_wireframe_series(seed_key, unit="currency", direction="up", periods=8):
    rng = random.Random(_seed_from_text(seed_key))
    unit_s = str(unit or "").lower()
    direction_s = str(direction or "").lower()

    if unit_s == "currency":
        start = rng.uniform(8_000_000, 90_000_000)
    elif unit_s == "count":
        start = rng.uniform(2_000, 80_000)
    elif unit_s == "percent":
        start = rng.uniform(8, 45)
    elif unit_s == "duration":
        start = rng.uniform(18, 160)
    elif unit_s == "score":
        start = rng.uniform(2.8, 4.4)
    else:
        start = rng.uniform(2_000, 50_000)

    if "down" in direction_s:
        drift = -rng.uniform(0.01, 0.05)
    elif "flat" in direction_s or "stable" in direction_s:
        drift = rng.uniform(-0.004, 0.004)
    else:
        drift = rng.uniform(0.01, 0.06)

    points = []
    value = start
    safe_periods = max(5, int(periods or 8))
    for i in range(safe_periods):
        noise = rng.uniform(-0.025, 0.025)
        seasonal = 0.012 * math.sin((i / max(1, safe_periods)) * math.pi * 2.0 + rng.uniform(0.0, 0.8))
        value = max(0.0001, value * (1.0 + drift + noise + seasonal))
        if unit_s == "percent":
            value = max(0.1, min(99.9, value))
        if unit_s == "score":
            value = max(1.0, min(5.0, value))
        points.append(round(value, 4))

    first = points[0]
    last = points[-1]
    delta_pct = 0.0 if first == 0 else ((last - first) / abs(first)) * 100.0
    return points, delta_pct


def _build_wireframe_payload(blueprint, prompt_text, requested_kpis=6):
    bp = blueprint if isinstance(blueprint, dict) else {}
    domain = str(bp.get("domain") or _wireframe_keyword_domain(prompt_text)).strip() or "Business Performance"
    title = str(bp.get("wireframe_title") or f"{domain} KPI Wireframe").strip()

    raw_kpis = bp.get("kpis")
    if not isinstance(raw_kpis, list) or not raw_kpis:
        fallback = _wireframe_fallback_blueprint(prompt_text, requested_kpis)
        raw_kpis = fallback["kpis"]
        if not title:
            title = fallback["wireframe_title"]

    limit = max(1, min(12, int(requested_kpis or 6)))
    raw_kpis = raw_kpis[:limit]

    month_labels = _wireframe_month_labels(8)
    kpis = []
    for i, item in enumerate(raw_kpis):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or f"KPI {i + 1}").strip() or f"KPI {i + 1}"
        unit = str(item.get("unit") or "count").strip().lower()
        direction = str(item.get("direction") or "up").strip().lower()
        intent = str(item.get("intent") or f"Track {label.lower()} over time.").strip()
        chart = str(item.get("chart") or "line").strip().lower()
        if chart not in {"line", "bar", "area"}:
            chart = "line"

        series, delta_pct = _generate_wireframe_series(
            seed_key=f"{domain}|{label}|{unit}|{direction}|{prompt_text}",
            unit=unit,
            direction=direction,
            periods=len(month_labels),
        )
        current = series[-1] if series else 0
        kpis.append(
            {
                "id": f"wkpi_{i + 1}",
                "label": label,
                "intent": intent,
                "unit": unit,
                "direction": direction,
                "chart": chart,
                "value": float(current),
                "value_display": _format_wireframe_value(current, unit),
                "delta_pct": round(float(delta_pct), 2),
                "sparkline": [float(v) for v in series],
                "periods": month_labels,
            }
        )

    assumptions = bp.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions:
        assumptions = [
            "Wireframe uses sample synthetic values, not live data.",
            "Domain and KPI list are inferred from user description.",
            "Use this for layout/metric alignment before data integration.",
        ]

    return {
        "domain": domain,
        "wireframe_title": title,
        "assumptions": [str(a) for a in assumptions[:6]],
        "kpis": kpis,
        "layout": {
            "cards_per_row": 3 if len(kpis) <= 6 else 4,
            "sections": [
                "KPI cards with trend chips",
                "Trend strip placeholders",
                "Comparative breakdown placeholders",
            ],
        },
    }


def _wireframe_normalize_chart_type(chart_type):
    t = str(chart_type or "bar").strip().lower()
    if t not in {"bar", "line", "area", "pie", "heatmap", "scatter"}:
        return "bar"
    return t


def _wireframe_find_kpi_by_label(kpis, name):
    target = str(name or "").strip().lower()
    if not target:
        return None
    for k in (kpis or []):
        label = str((k or {}).get("label") or "").strip().lower()
        if label == target:
            return k
    for k in (kpis or []):
        label = str((k or {}).get("label") or "").strip().lower()
        if target in label or label in target:
            return k
    return None


def _wireframe_default_chart_plans(kpis, chart_count=3):
    labels = [str((k or {}).get("label") or "") for k in (kpis or []) if k]
    plans = []
    if labels:
        plans.append({
            "title": f"{labels[0]} Trend",
            "type": "line",
            "source_kpis": [labels[0]],
            "xlabel": "Period",
            "ylabel": labels[0],
        })
    if len(labels) >= 2:
        plans.append({
            "title": "KPI Comparison",
            "type": "bar",
            "source_kpis": labels[: min(6, len(labels))],
            "xlabel": "KPI",
            "ylabel": "Value",
        })
    if len(labels) >= 2:
        plans.append({
            "title": "KPI Share",
            "type": "pie",
            "source_kpis": labels[: min(6, len(labels))],
            "xlabel": "KPI",
            "ylabel": "Share",
        })
    if not plans:
        plans = [{
            "title": "Metric Trend",
            "type": "line",
            "source_kpis": [],
            "xlabel": "Period",
            "ylabel": "Value",
        }]
    max_count = max(1, int(chart_count or 3))
    return plans[:max_count]


def _wireframe_build_chart_payload(plan, kpis, chart_id):
    chart_type = _wireframe_normalize_chart_type((plan or {}).get("type"))
    title = str((plan or {}).get("title") or "Wireframe Chart").strip() or "Wireframe Chart"
    xlabel = str((plan or {}).get("xlabel") or "").strip()
    ylabel = str((plan or {}).get("ylabel") or "").strip()

    src_names = (plan or {}).get("source_kpis") or []
    if not isinstance(src_names, list):
        src_names = [str(src_names)]

    selected = []
    for n in src_names:
        hit = _wireframe_find_kpi_by_label(kpis, n)
        if hit and hit not in selected:
            selected.append(hit)
    if not selected and kpis:
        selected = [kpis[0]]

    if chart_type in {"line", "area"}:
        primary = selected[0] if selected else {}
        x = list((primary or {}).get("periods") or [])
        y = [float(v) for v in ((primary or {}).get("sparkline") or [])]
        out = {
            "id": chart_id,
            "title": title,
            "type": "line" if chart_type == "line" else "area",
            "xlabel": xlabel or "Period",
            "ylabel": ylabel or str((primary or {}).get("label") or "Value"),
            "x": x,
            "y": y,
        }
        if len(selected) >= 2:
            sec = selected[1]
            out["series"] = [
                {"name": str((primary or {}).get("label") or "Series 1"), "x": x, "y": y},
                {
                    "name": str((sec or {}).get("label") or "Series 2"),
                    "x": list((sec or {}).get("periods") or x),
                    "y": [float(v) for v in ((sec or {}).get("sparkline") or [])],
                },
            ]
        return out

    if chart_type == "scatter":
        if len(selected) < 2 and len(kpis) >= 2:
            selected = [kpis[0], kpis[1]]
        a = selected[0] if selected else {}
        b = selected[1] if len(selected) > 1 else a
        ax = [float(v) for v in ((a or {}).get("sparkline") or [])]
        by = [float(v) for v in ((b or {}).get("sparkline") or [])]
        m = min(len(ax), len(by))
        return {
            "id": chart_id,
            "title": title,
            "type": "scatter",
            "xlabel": xlabel or str((a or {}).get("label") or "X"),
            "ylabel": ylabel or str((b or {}).get("label") or "Y"),
            "x": ax[:m],
            "y": by[:m],
        }

    if chart_type == "heatmap":
        if not selected:
            selected = (kpis or [])[:3]
        x = list((selected[0] or {}).get("periods") or [])
        y = [str((k or {}).get("label") or "KPI") for k in selected]
        z = []
        for k in selected:
            row = [float(v) for v in ((k or {}).get("sparkline") or [])]
            if len(row) < len(x):
                row = row + [row[-1] if row else 0.0] * (len(x) - len(row))
            z.append(row[: len(x)])
        return {
            "id": chart_id,
            "title": title,
            "type": "heatmap",
            "xlabel": xlabel or "Period",
            "ylabel": ylabel or "KPI",
            "x": x,
            "y": y,
            "z": z,
        }

    labels = [str((k or {}).get("label") or "KPI") for k in selected] if selected else []
    values = [float((k or {}).get("value") or 0.0) for k in selected] if selected else []

    if chart_type == "pie":
        return {
            "id": chart_id,
            "title": title,
            "type": "pie",
            "xlabel": xlabel or "KPI",
            "ylabel": ylabel or "Share",
            "x": labels,
            "y": values,
        }

    # Demo wireframe UX: a single-category bar looks like one giant block.
    # When only one KPI is selected, switch bar x/y to month-wise synthetic trend.
    if len(selected) == 1:
        solo = selected[0] or {}
        period_x = list((solo or {}).get("periods") or [])
        period_y = [float(v) for v in ((solo or {}).get("sparkline") or [])]
        if period_x and period_y:
            m = min(len(period_x), len(period_y))
            return {
                "id": chart_id,
                "title": title,
                "type": "bar",
                "xlabel": xlabel or "Period",
                "ylabel": ylabel or str((solo or {}).get("label") or "Value"),
                "x": period_x[:m],
                "y": period_y[:m],
            }

    return {
        "id": chart_id,
        "title": title,
        "type": "bar",
        "xlabel": xlabel or "KPI",
        "ylabel": ylabel or "Value",
        "x": labels,
        "y": values,
    }


def _wireframe_plan_charts_with_llm(
    domain,
    prompt,
    kpis,
    chart_count=3,
    logs=None,
    reference_image_block=None,
    prefer_image_layout=False,
):
    count = max(1, min(8, int(chart_count or 3)))
    kpi_list = [
        {
            "label": str((k or {}).get("label") or ""),
            "unit": str((k or {}).get("unit") or ""),
            "direction": str((k or {}).get("direction") or ""),
        }
        for k in (kpis or [])
    ]
    count_rule = (
        f"- Return as many visible charts from the reference image (up to {count})."
        if prefer_image_layout
        else f"- Return exactly {count} charts."
    )
    image_rule = (
        "- If reference image is present, replicate chart intent/types/titles visible in it."
        if prefer_image_layout
        else "- Prefer practical business visuals."
    )

    prompt_text = f"""
You are a BI wireframe designer. Based on domain + KPI list, propose chart placeholders.

Return ONLY JSON:
{{
  "charts": [
    {{
      "title": "Chart title",
      "type": "bar|line|area|pie|heatmap|scatter",
      "source_kpis": ["KPI label 1", "KPI label 2"],
      "xlabel": "x label",
      "ylabel": "y label"
    }}
  ]
}}

Rules:
- {count_rule}
- Use only source_kpis from the provided KPI labels.
- {image_rule}
- No SQL/code/markdown.

Domain: {domain}
User context: {prompt}
KPI list: {json.dumps(kpi_list, ensure_ascii=False)}
""".strip()

    content = [{"type": "text", "text": prompt_text}]
    if reference_image_block:
        content.append(reference_image_block)

    debug_target = None if reference_image_block else logs
    llm_raw, tokens_used = call_ai_with_retry(
        messages=[{"role": "user", "content": content if reference_image_block else prompt_text}],
        json_mode=True,
        retries=2,
        debug_logs=debug_target,
        context="Generate Wireframe Chart Plans",
    )
    if reference_image_block and logs is not None:
        logs.append("[LLM REQUEST] Generate Wireframe Chart Plans | image included")
        logs.append(f"[LLM RESPONSE] Generate Wireframe Chart Plans | tokens={tokens_used} | chars={len(str(llm_raw or ''))}")
    parsed = _parse_wireframe_llm_json(llm_raw)
    charts = (parsed or {}).get("charts") if isinstance(parsed, dict) else None
    if not isinstance(charts, list) or not charts:
        if logs is not None:
            logs.append("[WIRE] Using fallback chart plans")
        charts = _wireframe_default_chart_plans(kpis, chart_count=count)
    if prefer_image_layout:
        return charts[:count]
    return charts[:count]


def _wireframe_extract_explicit_targets(prompt_text):
    text = str(prompt_text or "").strip()
    lower = text.lower()
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }

    def _to_int(token):
        t = str(token or "").strip().lower()
        if t.isdigit():
            return int(t)
        return word_to_num.get(t)

    kpi_count = None
    chart_count = None

    for pat in [
        r"\b(?:exactly|around|about|roughly)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:kpis?|kpi cards?|metrics?)\b",
    ]:
        m = re.search(pat, lower)
        if m:
            v = _to_int(m.group(1))
            if v is not None:
                kpi_count = v
                break

    for pat in [
        r"\b(?:exactly|around|about|roughly)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:charts?|graphs?|plots?|visuals?)\b",
    ]:
        m = re.search(pat, lower)
        if m:
            v = _to_int(m.group(1))
            if v is not None:
                chart_count = v
                break

    kpi_labels = []
    kpi_list_match = re.search(
        r"\bkpi(?:\s+cards?)?\s+(?:for|like|including)\s+(.+?)(?:[.;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if kpi_list_match:
        raw = kpi_list_match.group(1)
        parts = re.split(r",|\band\b|&", raw, flags=re.IGNORECASE)
        for part in parts:
            label = str(part or "").strip(" .:-")
            if label and label.lower() not in {"etc", "others", "other"}:
                kpi_labels.append(label)
        if len(kpi_labels) >= 2:
            kpi_count = len(kpi_labels) if kpi_count is None else kpi_count

    chart_types = []
    for t in ["bar", "line", "area", "pie", "heatmap", "scatter"]:
        if re.search(rf"\b{re.escape(t)}(?:\s+chart)?s?\b", lower):
            chart_types.append(t)

    has_explicit_layout = bool(
        kpi_count is not None or chart_count is not None or kpi_labels or chart_types
    )
    return {
        "kpi_count": kpi_count,
        "chart_count": chart_count,
        "kpi_labels": kpi_labels[:12],
        "chart_types": chart_types[:8],
        "has_explicit_layout": has_explicit_layout,
    }


def generate_wireframe_from_prompt(
    description,
    kpi_count=6,
    chart_count=3,
    reference_image_b64=None,
    reference_mime="image/png",
):
    llm_logs = []
    prompt = str(description or "").strip()
    if not prompt:
        raise ValueError("Description is required")

    try:
        requested_kpis = max(3, min(8, int(kpi_count or 6)))
    except Exception:
        requested_kpis = 6
    try:
        requested_charts = max(0, min(8, int(chart_count or 3)))
    except Exception:
        requested_charts = 3

    explicit_targets = _wireframe_extract_explicit_targets(prompt)
    explicit_kpi_labels = [str(x).strip() for x in (explicit_targets.get("kpi_labels") or []) if str(x).strip()]
    explicit_chart_types = [
        _wireframe_normalize_chart_type(t)
        for t in (explicit_targets.get("chart_types") or [])
        if str(t).strip()
    ]
    explicit_kpi_count = explicit_targets.get("kpi_count")
    explicit_chart_count = explicit_targets.get("chart_count")
    text_priority_mode = bool(
        explicit_kpi_labels
        or explicit_chart_types
        or (explicit_kpi_count is not None and explicit_chart_count is not None)
    )

    if explicit_kpi_count is not None:
        requested_kpis = max(1, min(8, int(explicit_kpi_count)))
    elif explicit_kpi_labels:
        requested_kpis = max(1, min(8, len(explicit_kpi_labels)))

    if explicit_chart_count is not None:
        requested_charts = max(0, min(8, int(explicit_chart_count)))
    elif explicit_chart_types:
        requested_charts = max(0, min(8, len(explicit_chart_types)))

    image_block = None
    b64 = str(reference_image_b64 or "").strip()
    if b64:
        mime = str(reference_mime or "image/png").strip()
        if not mime.startswith("image/"):
            mime = "image/png"
        image_block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        llm_logs.append("[WIRE] Reference image attached")

    if text_priority_mode:
        llm_logs.append(
            f"[WIRE] Text directives applied: kpis={requested_kpis}, charts={requested_charts}"
        )
    elif image_block:
        llm_logs.append("[WIRE] Image-first layout mode active")

    if text_priority_mode:
        priority_rule = "- Text directives are strict priority. If image conflicts with explicit text, follow text."
    elif image_block:
        priority_rule = "- If reference image is present, treat image as primary truth for visible KPI cards and chart composition."
    else:
        priority_rule = "- Infer KPI/chart layout from text requirements."

    explicit_rules = []
    if explicit_kpi_count is not None:
        explicit_rules.append(f"Explicit KPI count from text: {requested_kpis}")
    if explicit_kpi_labels:
        explicit_rules.append(f"Explicit KPI labels from text: {', '.join(explicit_kpi_labels[:8])}")
    if explicit_chart_count is not None:
        explicit_rules.append(f"Explicit chart count from text: {requested_charts}")
    if explicit_chart_types:
        explicit_rules.append(f"Explicit chart types from text (order): {', '.join(explicit_chart_types[:8])}")
    if not explicit_rules:
        explicit_rules.append("No strict text directives detected.")
    explicit_rule_block = "\n".join([f"- {r}" for r in explicit_rules])

    prompt_text = f"""
You are a senior BI wireframe designer.
Given user requirements (no real dataset), infer the best business domain and draft KPI wireframe metadata.

Return ONLY JSON with this shape:
{{
  "domain": "short domain name",
  "wireframe_title": "dashboard/wireframe title",
  "assumptions": ["assumption 1", "assumption 2"],
  "kpis": [
    {{
      "label": "KPI label",
      "unit": "currency|count|percent|duration|score",
      "direction": "up|down|stable",
      "chart": "line|bar|area",
      "intent": "why this KPI matters"
    }}
  ],
  "charts": [
    {{
      "title": "Chart title",
      "type": "bar|line|area|pie|heatmap|scatter",
      "source_kpis": ["KPI label 1", "KPI label 2"],
      "xlabel": "x label",
      "ylabel": "y label"
    }}
  ]
}}

Rules:
- {priority_rule}
- Propose exactly {requested_kpis} KPIs.
- Propose exactly {requested_charts} charts.
- Preserve explicit KPI labels/chart types from text when provided.
- Keep domain specific and practical.
- KPI names should be business-friendly and non-technical.
- Do not include SQL, code, or markdown.

Text directives:
{explicit_rule_block}

User requirement text:
{prompt}
""".strip()

    if image_block:
        content = [
            {"type": "text", "text": prompt_text},
            image_block,
        ]
    else:
        content = prompt_text

    llm_raw = None
    blueprint = None
    try:
        debug_target = None if image_block else llm_logs
        llm_raw, tokens_used = call_ai_with_retry(
            messages=[{"role": "user", "content": content}],
            json_mode=True,
            retries=2,
            debug_logs=debug_target,
            context="Generate Wireframe Blueprint",
        )
        if image_block:
            redacted_prompt = _redact_sensitive_text(prompt_text)
            preview_chars = 6000
            prompt_preview = redacted_prompt[:preview_chars]
            if len(redacted_prompt) > preview_chars:
                prompt_preview += f"\n... [TRUNCATED {len(redacted_prompt) - preview_chars} chars]"
            llm_logs.append(
                f"[LLM REQUEST] Generate Wireframe Blueprint | image included | text_chars={len(redacted_prompt)}\n"
                f"[user]\n{prompt_preview}"
            )
            llm_logs.append(
                f"[LLM RESPONSE] Generate Wireframe Blueprint | tokens={tokens_used} | chars={len(str(llm_raw or ''))}"
            )
        blueprint = _parse_wireframe_llm_json(llm_raw)
    except Exception as e:
        llm_logs.append(f"[WIRE][WARN] LLM blueprint failed: {str(e)}")
        blueprint = None

    if not isinstance(blueprint, dict):
        llm_logs.append("[WIRE] Using heuristic fallback blueprint")
        blueprint = _wireframe_fallback_blueprint(prompt, requested_kpis)

    if explicit_kpi_labels:
        deduped_labels = []
        seen = set()
        for label in explicit_kpi_labels:
            clean = re.sub(r"\s+", " ", str(label or "")).strip(" .:-")
            key = clean.lower()
            if clean and key not in seen:
                deduped_labels.append(clean)
                seen.add(key)
        base_kpis = blueprint.get("kpis") if isinstance(blueprint.get("kpis"), list) else []
        enforced_kpis = []
        for idx, label in enumerate(deduped_labels[:requested_kpis]):
            src = base_kpis[idx] if idx < len(base_kpis) and isinstance(base_kpis[idx], dict) else {}
            chart_hint = str(src.get("chart") or "line").strip().lower()
            if chart_hint not in {"line", "bar", "area"}:
                chart_hint = "line"
            enforced_kpis.append(
                {
                    "label": label,
                    "unit": str(src.get("unit") or "count").strip().lower() or "count",
                    "direction": str(src.get("direction") or "up").strip().lower() or "up",
                    "chart": chart_hint,
                    "intent": str(src.get("intent") or f"Track {label.lower()} over time.").strip(),
                }
            )
        for idx in range(len(enforced_kpis), requested_kpis):
            src = base_kpis[idx] if idx < len(base_kpis) and isinstance(base_kpis[idx], dict) else {}
            label = str(src.get("label") or f"KPI {idx + 1}").strip() or f"KPI {idx + 1}"
            chart_hint = str(src.get("chart") or "line").strip().lower()
            if chart_hint not in {"line", "bar", "area"}:
                chart_hint = "line"
            enforced_kpis.append(
                {
                    "label": label,
                    "unit": str(src.get("unit") or "count").strip().lower() or "count",
                    "direction": str(src.get("direction") or "up").strip().lower() or "up",
                    "chart": chart_hint,
                    "intent": str(src.get("intent") or f"Track {label.lower()} over time.").strip(),
                }
            )
        blueprint["kpis"] = enforced_kpis[:requested_kpis]
        llm_logs.append(f"[WIRE] Enforced KPI labels from text: {len(deduped_labels[:requested_kpis])}")

    if image_block and (not text_priority_mode) and isinstance(blueprint.get("kpis"), list) and blueprint.get("kpis"):
        inferred_kpis = len(blueprint.get("kpis") or [])
        requested_kpis = max(1, min(12, inferred_kpis))
        llm_logs.append(f"[WIRE] Image-first KPI inference applied: kpis={requested_kpis}")

    payload = _build_wireframe_payload(blueprint, prompt, requested_kpis=requested_kpis)
    blueprint_charts = blueprint.get("charts") if isinstance(blueprint, dict) else None
    chart_target = requested_charts
    chart_plans = []
    prefer_image_layout = bool(image_block and not text_priority_mode)
    if chart_target <= 0:
        llm_logs.append("[WIRE] Chart count set to 0; skipping chart generation")
    elif prefer_image_layout and isinstance(blueprint_charts, list) and blueprint_charts:
        chart_plans = blueprint_charts[: max(1, min(8, len(blueprint_charts)))]
        llm_logs.append(f"[WIRE] Image-first chart inference applied: charts={len(chart_plans)}")
    else:
        image_chart_cap = 8 if prefer_image_layout else chart_target
        chart_plans = _wireframe_plan_charts_with_llm(
            domain=payload.get("domain", ""),
            prompt=prompt,
            kpis=payload.get("kpis", []),
            chart_count=image_chart_cap,
            logs=llm_logs,
            reference_image_block=image_block if (image_block and not text_priority_mode) else None,
            prefer_image_layout=prefer_image_layout,
        )
    if explicit_chart_types:
        first_label = str((payload.get("kpis") or [{}])[0].get("label") or "KPI") if (payload.get("kpis") or []) else "KPI"
        for idx, forced_type in enumerate(explicit_chart_types[:requested_charts]):
            if idx < len(chart_plans) and isinstance(chart_plans[idx], dict):
                chart_plans[idx]["type"] = forced_type
                if not chart_plans[idx].get("source_kpis"):
                    chart_plans[idx]["source_kpis"] = [first_label]
            else:
                chart_plans.append(
                    {
                        "title": f"{forced_type.title()} Chart {idx + 1}",
                        "type": forced_type,
                        "source_kpis": [first_label],
                        "xlabel": "Period" if forced_type in {"line", "area", "heatmap"} else "Category",
                        "ylabel": "Value",
                    }
                )
        llm_logs.append(f"[WIRE] Enforced chart types from text: {', '.join(explicit_chart_types[:requested_charts])}")
    if not prefer_image_layout:
        chart_plans = (chart_plans or [])[: max(0, min(8, requested_charts))]

    charts = []
    for idx, plan in enumerate(chart_plans):
        charts.append(_wireframe_build_chart_payload(plan, payload.get("kpis", []), chart_id=f"wchart_{idx+1}"))

    payload["charts"] = charts
    payload["chart_count"] = len(charts)
    payload["kpi_count"] = len(payload.get("kpis", []))
    payload["input"] = {
        "description": prompt,
        "kpi_count": requested_kpis,
        "chart_count": requested_charts,
        "has_reference_image": bool(image_block),
    }
    payload["logs"] = llm_logs
    return payload


def generate_wireframe_artifact_from_prompt(user_prompt, current_payload, artifact_hint="auto"):
    prompt = str(user_prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")

    state = current_payload if isinstance(current_payload, dict) else {}
    kpis = state.get("kpis", [])
    charts = state.get("charts", [])
    if not isinstance(kpis, list):
        kpis = []
    if not isinstance(charts, list):
        charts = []
    domain = str(state.get("domain") or "Business Performance")

    logs = _LiveLogBuffer(context="Configure KPI")
    hint = str(artifact_hint or "auto").strip().lower()
    if hint not in {"auto", "kpi", "chart"}:
        hint = "auto"

    kpi_meta = [{"label": str((k or {}).get("label") or ""), "unit": str((k or {}).get("unit") or "")} for k in kpis]
    chart_meta = [{"title": str((c or {}).get("title") or ""), "type": str((c or {}).get("type") or "")} for c in charts]

    def _next_unique_artifact_id(prefix, items):
        used = set()
        max_num = 0
        for item in (items or []):
            raw_id = str((item or {}).get("id") or "").strip()
            if not raw_id:
                continue
            used.add(raw_id)
            m = re.match(rf"^{re.escape(prefix)}_(\d+)$", raw_id)
            if m:
                try:
                    max_num = max(max_num, int(m.group(1)))
                except Exception:
                    pass
        candidate = max_num + 1 if max_num > 0 else (len(items or []) + 1)
        while f"{prefix}_{candidate}" in used:
            candidate += 1
        return f"{prefix}_{candidate}"

    prompt_text = f"""
You are a BI wireframe assistant.
Decide whether user wants a new KPI card or a new chart, then propose one artifact.

Return ONLY JSON:
{{
  "artifact_type": "kpi|chart",
  "kpi": {{
    "label": "KPI label",
    "unit": "currency|count|percent|duration|score",
    "direction": "up|down|stable",
    "chart": "line|bar|area",
    "intent": "why this KPI matters"
  }},
  "chart": {{
    "title": "Chart title",
    "type": "bar|line|area|pie|heatmap|scatter",
    "source_kpis": ["KPI label 1", "KPI label 2"],
    "xlabel": "x label",
    "ylabel": "y label"
  }}
}}

Rules:
- Respect artifact hint when not auto: {hint}
- Propose exactly one artifact.
- Use source_kpis from existing KPI labels only.

Domain: {domain}
Existing KPIs: {json.dumps(kpi_meta, ensure_ascii=False)}
Existing Charts: {json.dumps(chart_meta, ensure_ascii=False)}
User request: {prompt}
""".strip()

    llm_raw, _ = call_ai_with_retry(
        messages=[{"role": "user", "content": prompt_text}],
        json_mode=True,
        retries=2,
        debug_logs=logs,
        context="Generate Wireframe Artifact",
    )
    parsed = _parse_wireframe_llm_json(llm_raw) or {}

    artifact_type = str(parsed.get("artifact_type") or "").strip().lower()
    if hint in {"kpi", "chart"}:
        artifact_type = hint
    if artifact_type not in {"kpi", "chart"}:
        low = prompt.lower()
        artifact_type = "chart" if any(w in low for w in ["chart", "graph", "plot", "visual"]) else "kpi"

    if artifact_type == "kpi":
        kpi_plan = parsed.get("kpi") if isinstance(parsed.get("kpi"), dict) else {}
        if not kpi_plan:
            kpi_plan = {
                "label": "New KPI",
                "unit": "count",
                "direction": "up",
                "chart": "line",
                "intent": f"Track {prompt.lower()}",
            }
        kpi_id = _next_unique_artifact_id("wkpi", kpis)
        month_labels = _wireframe_month_labels(8)
        label = str(kpi_plan.get("label") or "New KPI")
        unit = str(kpi_plan.get("unit") or "count").lower()
        direction = str(kpi_plan.get("direction") or "up").lower()
        series, delta_pct = _generate_wireframe_series(
            seed_key=f"{domain}|{label}|{unit}|{direction}|{prompt}|artifact",
            unit=unit,
            direction=direction,
            periods=len(month_labels),
        )
        value = series[-1] if series else 0
        kpi_payload = {
            "id": kpi_id,
            "label": label,
            "intent": str(kpi_plan.get("intent") or f"Track {label.lower()}"),
            "unit": unit,
            "direction": direction,
            "chart": str(kpi_plan.get("chart") or "line").lower(),
            "value": float(value),
            "value_display": _format_wireframe_value(value, unit),
            "delta_pct": round(float(delta_pct), 2),
            "sparkline": [float(v) for v in series],
            "periods": month_labels,
        }
        return {
            "artifact_type": "kpi",
            "kpi": kpi_payload,
            "logs": logs,
        }

    chart_plan = parsed.get("chart") if isinstance(parsed.get("chart"), dict) else {}
    if not chart_plan:
        defaults = _wireframe_default_chart_plans(kpis, chart_count=1)
        chart_plan = defaults[0] if defaults else {"title": "New Chart", "type": "bar", "source_kpis": []}
    chart_id = _next_unique_artifact_id("wchart", charts)
    chart_payload = _wireframe_build_chart_payload(chart_plan, kpis, chart_id=chart_id)
    return {
        "artifact_type": "chart",
        "chart": chart_payload,
        "logs": logs,
    }

from django.http import JsonResponse
def _extract_llm_logs(lines):
    if not isinstance(lines, list):
        return []
    include_sql = str(os.getenv("DATABRICKS_LOG_SQL_TEXT") or "").strip().lower() in {"1", "true", "yes", "on"}
    keep = []
    for line in lines:
        s = str(line or '')
        if ('[LLM REQUEST]' in s) or ('[LLM RESPONSE]' in s) or ('[LLM ERROR]' in s) or (include_sql and s.startswith('[SQL]')):
            keep.append(s)
    return keep


def _extract_configure_map_minimal_logs(lines):
    if not isinstance(lines, list):
        return []
    keep = []
    for line in lines:
        s = str(line or "")
        if not s:
            continue
        if (
            "[LLM REQUEST] Configure Chart Input" in s
            or "[LLM RESPONSE] Configure SQL" in s
            or s.startswith("[FALLBACK]")
            or s.startswith("[ERROR]")
            or "[LLM ERROR]" in s
            or s.startswith("[WARN]")
            or "[MAP] Loaded map data" in s
        ):
            keep.append(s)
    return keep[-80:]


def _extract_dashboard_generation_logs(lines):
    if not isinstance(lines, list):
        return []
    prefixes = (
        "[STEP]",
        "[PERF]",
        "[SOURCE]",
        "[FILTER]",
        "[CACHE]",
        "[WARN]",
        "[ERROR]",
        "[SECURITY]",
        "[FALLBACK]",
        "[GUARD]",
        "[DEDUP]",
        "[SQL]",
    )
    keep = []
    for line in lines:
        s = str(line or "")
        if s.startswith(prefixes) or ("[LLM REQUEST]" in s) or ("[LLM RESPONSE]" in s) or ("[LLM ERROR]" in s):
            keep.append(s)
    return keep[-2000:]


def _extract_filter_refresh_logs(lines):
    if not isinstance(lines, list):
        return []
    prefixes = (
        "[PERF]",
        "[SOURCE]",
        "[FILTER]",
        "[CACHE]",
        "[WARN]",
        "[ERROR]",
        "[SECURITY]",
        "[FALLBACK]",
        "[GUARD]",
        "[SQL]",
    )
    keep = []
    for line in lines:
        s = str(line or "")
        if s.startswith(prefixes):
            keep.append(s)
    return keep


def _sanitize_json_payload(value):
    if isinstance(value, dict):
        return {k: _sanitize_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_payload(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return _sanitize_json_payload(value.item())
        except Exception:
            return value
    return value


def _safe_json_response(payload, status=200):
    cleaned = _sanitize_json_payload(payload)
    return JsonResponse(cleaned, status=status, json_dumps_params={"allow_nan": False})


def _parse_request_payload(request):
    content_type = str(request.META.get("CONTENT_TYPE") or "").lower()
    if "application/json" in content_type:
        try:
            parsed = json.loads((request.body or b"{}").decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return request.POST


def dashboard_configure_sql(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    payload = _parse_request_payload(request)
    action = str(payload.get("action") or payload.get("mode") or "generate").strip().lower()
    kb_module_name = str(payload.get("kb_module_name") or "").strip() or None
    filters_json = (
        payload.get("filters")
        or payload.get("filters_json")
        or request.session.get("active_filters_json")
        or "{}"
    )

    try:
        if action in {"metadata", "schema", "init"}:
            logs = []
            resolved_kb_module = _resolve_kb_module_name(kb_module_name)
            kb_data = _fetch_knowledge_base_from_db(module_name=resolved_kb_module, logs=logs)
            catalog = _build_manual_config_table_catalog(kb_data)
            return _safe_json_response(
                {
                    "module": resolved_kb_module,
                    "tables": catalog.get("tables", []),
                    "relationships": catalog.get("relationships", []),
                    "metrics": catalog.get("metrics", []),
                    "logs": _extract_llm_logs(logs),
                    "data_mode": "databricks",
                }
            )

        config_payload = {
            "x_table": payload.get("x_table"),
            "x_column": payload.get("x_column"),
            "y_table": payload.get("y_table"),
            "y_column": payload.get("y_column"),
            "aggregation": payload.get("aggregation"),
            "chart_type": payload.get("chart_type"),
            "metric_name": payload.get("metric_name"),
            "metric_formula": payload.get("metric_formula"),
            "metric_axis": payload.get("metric_axis"),
            "dimension_axis": payload.get("dimension_axis"),
            "table_name": payload.get("table_name"),
            "column_name": payload.get("column_name"),
        }

        if action in {"generate_kpi", "kpi"}:
            result = generate_manual_configured_kpi_databricks(
                config_payload,
                active_filters_json=filters_json,
                kb_module_name=kb_module_name,
            )
        else:
            result = generate_manual_configured_chart_databricks(
                config_payload,
                active_filters_json=filters_json,
                kb_module_name=kb_module_name,
            )
        result_chart = result.get("chart") if isinstance(result, dict) else {}
        result_type = str((result_chart or {}).get("type") or config_payload.get("chart_type") or "").strip().lower()
        if action in {"generate_kpi", "kpi"}:
            result["logs"] = _extract_llm_logs(result.get("logs", []))
        elif result_type == "india_map":
            result["logs"] = _extract_configure_map_minimal_logs(result.get("logs", []))
        else:
            result["logs"] = _extract_llm_logs(result.get("logs", []))
        return _safe_json_response(result)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


def _json_text(value, default="{}"):
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return default
    text = str(value or "").strip()
    if not text:
        return default
    try:
        parsed = json.loads(text)
        return json.dumps(parsed)
    except Exception:
        return default


def _json_value(value, default=None):
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return {} if default is None else default
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, (dict, list)) else ({} if default is None else default)
    except Exception:
        return {} if default is None else default



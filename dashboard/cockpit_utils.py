import io
import math
import re
import os
from calendar import monthrange
from datetime import datetime
import json

import numpy as np
import pandas as pd
from .utils import call_ai_with_retry
from .databricks_client import (
    get_databricks_connection,
    fetch_dataframe,
    DatabricksConfigError,
)

try:
    import duckdb
except Exception:
    duckdb = None


MAX_UPLOAD_ROWS = 250000
DEFAULT_COCKPIT_INVOICE_TABLE = "llm_test.llm.invoice_template"
DEFAULT_COCKPIT_TARGET_TABLE = "llm_test.llm.target_template"


def _sql_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def _normalize_cockpit_df(df):
    if df is None:
        return None
    out = df.copy()
    if "_date" in out.columns:
        out["_date"] = pd.to_datetime(out["_date"], errors="coerce")
    for c in [
        "_account_group",
        "_bcra_dairy",
        "_brand",
        "_region",
        "_business_channel",
        "_category",
        "_outlet",
        "_productive_flag",
    ]:
        if c in out.columns:
            out[c] = out[c].astype(str).replace({"nan": "", "None": ""}).fillna("")
            out[c] = out[c].str.strip()
    for c in [
        "_value_metric",
        "_volume_metric",
        "_ly_value_metric",
        "_ly_volume_metric",
        "_msp_metric",
        "_mtd_metric",
        "_eco_thousands",
        "_le_metric",
        "_daily_value_metric",
        "_sly_pct_col",
        "_seq_pct_col",
        "_working_days_col",
        "_days_lapsed_col",
        "_days_left_col",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


def _safe_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except Exception:
        return 0.0


def _pct_delta(curr, prev):
    c = _safe_float(curr)
    p = _safe_float(prev)
    if abs(p) < 1e-9:
        return 0.0
    return ((c - p) / abs(p)) * 100.0


def _find_col(columns, aliases):
    norm_map = {_norm(c): c for c in columns}
    for a in aliases:
        key = _norm(a)
        if key in norm_map:
            return norm_map[key]
    for c in columns:
        n = _norm(c)
        for a in aliases:
            if _norm(a) and _norm(a) in n:
                return c
    return None


def _series_as_str(series, default="All"):
    out = series.astype(str).replace({"nan": default, "None": default}).fillna(default).str.strip()
    out = out.replace("", default)
    return out


def _choose_numeric_columns(df):
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def _quote_db_ident(name):
    return "`" + str(name or "").replace("`", "``") + "`"


def _quote_db_table_fqn(table_name):
    parts = [p.strip() for p in str(table_name or "").split(".") if p and str(p).strip()]
    if not parts:
        raise ValueError(f"Invalid table name: {table_name}")
    return ".".join(_quote_db_ident(p) for p in parts)


def _prepare_cockpit_frame(df):
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Dataset has too many rows ({len(df)}). Limit is {MAX_UPLOAD_ROWS}.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = _find_col(
        df.columns,
        [
            "date",
            "billdate",
            "invoice date",
            "invoice_date",
            "transaction_date",
            "posting_date",
            "day",
        ],
    )
    account_col = _find_col(df.columns, ["account_group", "account group", "customer type", "customer_group"])
    bcra_col = _find_col(df.columns, ["bcra & dairy", "bcra", "dairy", "segment", "division"])
    brand_col = _find_col(df.columns, ["brand_filter", "brand", "product_brand", "subbrand"])
    region_col = _find_col(df.columns, ["region", "region_name", "zone", "state"])
    channel_col = _find_col(df.columns, ["business channel", "business_channel", "channel"])
    category_col = _find_col(df.columns, ["category", "category_name", "product_category"])
    outlet_col = _find_col(df.columns, ["outlet", "retailer", "customer", "sold_to_code", "customer_name", "store"])
    productive_col = _find_col(df.columns, ["productive", "productive_flag", "is_productive", "visit_type"])

    numeric_cols = _choose_numeric_columns(df)

    msp_col = _find_col(numeric_cols, ["msp_value_cr", "msp", "msp value"])
    mtd_col = _find_col(numeric_cols, ["mtd_value_cr", "cmtd", "mtd", "mtd value"])
    eco_col = _find_col(numeric_cols, ["eco_thousands", "eco", "eco thousands"])
    le_col = _find_col(numeric_cols, ["le", "latest estimate", "le_in_cr", "le in cr"])
    sly_col = _find_col(numeric_cols, ["sly_pct", "sly%", "sly pct"])
    seq_col = _find_col(numeric_cols, ["seq_pct", "seq%", "seq pct"])
    working_days_col = _find_col(numeric_cols, ["working_days_month", "working days", "working_days"])
    days_lapsed_col = _find_col(numeric_cols, ["days_lapsed", "days lapsed"])
    days_left_col = _find_col(numeric_cols, ["days_left", "days left"])
    daily_value_col = _find_col(numeric_cols, ["daily_value_cr", "daily value", "daily sales"])

    value_col = _find_col(
        numeric_cols,
        [
            "netamount",
            "gross_value",
            "value",
            "sales",
            "amount",
            "daily_value_cr",
            "revenue",
            "mtd_value_cr",
            "cmtd",
            "mtd",
        ],
    )
    volume_col = _find_col(
        numeric_cols,
        [
            "volume_units",
            "invoicequantity",
            "quantity",
            "qty",
            "volume",
            "units",
            "packs",
        ],
    )
    if not value_col and numeric_cols:
        value_col = numeric_cols[0]
    if not volume_col and len(numeric_cols) > 1:
        volume_col = numeric_cols[1]
    if not volume_col:
        volume_col = value_col

    ly_value_col = _find_col(
        numeric_cols,
        [
            "lymtd",
            "ly_mtd",
            "ly mtd",
            "last year mtd",
            "prev year mtd",
            "last_year_mtd",
            "ly_value",
        ],
    )
    ly_volume_col = _find_col(
        numeric_cols,
        [
            "ly volume",
            "ly_volume",
            "ly qty",
            "ly quantity",
            "ly_quantity",
            "ly_invoicequantity",
        ],
    )

    df["_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    df["_account_group"] = _series_as_str(df[account_col], "All") if account_col else "All"
    df["_bcra_dairy"] = _series_as_str(df[bcra_col], "All") if bcra_col else "All"
    df["_brand"] = _series_as_str(df[brand_col], "All") if brand_col else "All"
    df["_region"] = _series_as_str(df[region_col], "Unknown") if region_col else "Unknown"
    df["_business_channel"] = (
        _series_as_str(df[channel_col], "Unknown") if channel_col else "Unknown"
    )
    df["_category"] = _series_as_str(df[category_col], "Unknown") if category_col else "Unknown"
    df["_outlet"] = _series_as_str(df[outlet_col], "Unknown Outlet") if outlet_col else "Unknown Outlet"

    df["_value_metric"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0) if value_col else 0.0
    df["_volume_metric"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0) if volume_col else 0.0
    df["_ly_value_metric"] = pd.to_numeric(df[ly_value_col], errors="coerce").fillna(0.0) if ly_value_col else 0.0
    df["_ly_volume_metric"] = pd.to_numeric(df[ly_volume_col], errors="coerce").fillna(0.0) if ly_volume_col else 0.0

    df["_msp_metric"] = (
        pd.to_numeric(df[msp_col], errors="coerce").fillna(0.0)
        if msp_col
        else pd.to_numeric(df[mtd_col], errors="coerce").fillna(0.0)
        if mtd_col
        else pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        if value_col
        else 0.0
    )
    df["_mtd_metric"] = (
        pd.to_numeric(df[mtd_col], errors="coerce").fillna(0.0)
        if mtd_col
        else pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        if value_col
        else 0.0
    )
    df["_eco_thousands"] = (
        pd.to_numeric(df[eco_col], errors="coerce").fillna(0.0)
        if eco_col
        else (pd.to_numeric(df[mtd_col], errors="coerce").fillna(0.0) * 1000.0)
        if mtd_col
        else 0.0
    )
    df["_le_metric"] = (
        pd.to_numeric(df[le_col], errors="coerce").fillna(0.0)
        if le_col
        else (df["_eco_thousands"] / 1000.0)
    )
    df["_daily_value_metric"] = (
        pd.to_numeric(df[daily_value_col], errors="coerce").fillna(0.0)
        if daily_value_col
        else df["_value_metric"]
    )
    df["_sly_pct_col"] = pd.to_numeric(df[sly_col], errors="coerce").fillna(0.0) if sly_col else 0.0
    df["_seq_pct_col"] = pd.to_numeric(df[seq_col], errors="coerce").fillna(0.0) if seq_col else 0.0
    df["_working_days_col"] = (
        pd.to_numeric(df[working_days_col], errors="coerce").fillna(0.0) if working_days_col else 0.0
    )
    df["_days_lapsed_col"] = (
        pd.to_numeric(df[days_lapsed_col], errors="coerce").fillna(0.0) if days_lapsed_col else 0.0
    )
    df["_days_left_col"] = (
        pd.to_numeric(df[days_left_col], errors="coerce").fillna(0.0) if days_left_col else 0.0
    )

    if productive_col:
        prod = _series_as_str(df[productive_col], "Productive").str.lower()
        df["_productive_flag"] = prod.apply(
            lambda x: "Non Productive"
            if any(t in x for t in ["non", "no", "0", "n", "false"])
            else "Productive"
        )
    else:
        q = df["_value_metric"].quantile(0.35)
        df["_productive_flag"] = np.where(df["_value_metric"] >= q, "Productive", "Non Productive")

    meta = {
        "date_col": date_col or "",
        "value_col": value_col or "",
        "volume_col": volume_col or "",
        "ly_value_col": ly_value_col or "",
        "ly_volume_col": ly_volume_col or "",
        "msp_col": msp_col or "",
        "mtd_col": mtd_col or "",
        "eco_col": eco_col or "",
        "le_col": le_col or "",
        "sly_col": sly_col or "",
        "seq_col": seq_col or "",
        "daily_value_col": daily_value_col or "",
        "working_days_col": working_days_col or "",
        "days_lapsed_col": days_lapsed_col or "",
        "days_left_col": days_left_col or "",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    return _normalize_cockpit_df(df), meta


def load_and_prepare_cockpit_excel(file_blob):
    try:
        df = pd.read_excel(io.BytesIO(file_blob))
    except ImportError as e:
        raise ValueError(
            "Excel support requires openpyxl. Install with: pip install openpyxl"
        ) from e
    except Exception as e:
        raise ValueError(f"Unable to read Excel file: {str(e)}") from e
    return _prepare_cockpit_frame(df)


def load_and_prepare_cockpit_databricks(
    invoice_table=DEFAULT_COCKPIT_INVOICE_TABLE,
    target_table=DEFAULT_COCKPIT_TARGET_TABLE,
):
    invoice_fqn = _quote_db_table_fqn(invoice_table)
    target_fqn = _quote_db_table_fqn(target_table)

    try:
        connection = get_databricks_connection()
    except DatabricksConfigError as e:
        raise ValueError(str(e)) from e

    try:
        max_sql = f"SELECT MAX(CAST({_quote_db_ident('Date')} AS DATE)) AS max_date FROM {invoice_fqn}"
        print(f"[COCKPIT SQL] {max_sql}", flush=True)
        max_df = fetch_dataframe(connection, max_sql, readonly=True)
        print(f"[COCKPIT RESULT] rows={0 if max_df is None else len(max_df)}", flush=True)
    except Exception as e:
        raise ValueError(f"Unable to read Databricks cockpit tables: {str(e)}") from e
    finally:
        try:
            connection.close()
        except Exception:
            pass

    if max_df is None or max_df.empty:
        raise ValueError("No data returned from Databricks cockpit tables.")

    max_raw = max_df.iloc[0, 0] if not max_df.empty else None
    max_ts = pd.to_datetime(max_raw, errors="coerce")
    if pd.isna(max_ts):
        raise ValueError("Could not resolve max Date from invoice table.")

    max_ts = pd.Timestamp(max_ts).normalize()
    curr_start = max_ts.replace(day=1)
    working_days_val = int(monthrange(max_ts.year, max_ts.month)[1])
    days_lapsed_val = int(max_ts.day)
    days_left_val = max(0, working_days_val - days_lapsed_val)

    filter_options = {
        "account_group": [],
        "brand": [],
        "category": [],
        "months": [],
    }
    try:
        sql_channel = (
            f"SELECT DISTINCT TRIM(CAST({_quote_db_ident('Channel')} AS STRING)) AS v "
            f"FROM {invoice_fqn} WHERE {_quote_db_ident('Channel')} IS NOT NULL "
            f"AND TRIM(CAST({_quote_db_ident('Channel')} AS STRING)) <> '' ORDER BY 1"
        )
        print(f"[COCKPIT SQL] {sql_channel}", flush=True)
        ch_df = fetch_dataframe(connection, sql_channel, readonly=True)
        print(f"[COCKPIT RESULT] rows={0 if ch_df is None else len(ch_df)}", flush=True)

        sql_brand = (
            f"SELECT DISTINCT TRIM(CAST({_quote_db_ident('Brand')} AS STRING)) AS v "
            f"FROM {invoice_fqn} WHERE {_quote_db_ident('Brand')} IS NOT NULL "
            f"AND TRIM(CAST({_quote_db_ident('Brand')} AS STRING)) <> '' ORDER BY 1"
        )
        print(f"[COCKPIT SQL] {sql_brand}", flush=True)
        br_df = fetch_dataframe(connection, sql_brand, readonly=True)
        print(f"[COCKPIT RESULT] rows={0 if br_df is None else len(br_df)}", flush=True)

        sql_category = (
            f"SELECT DISTINCT TRIM(CAST({_quote_db_ident('Category')} AS STRING)) AS v "
            f"FROM {invoice_fqn} WHERE {_quote_db_ident('Category')} IS NOT NULL "
            f"AND TRIM(CAST({_quote_db_ident('Category')} AS STRING)) <> '' ORDER BY 1"
        )
        print(f"[COCKPIT SQL] {sql_category}", flush=True)
        cat_df = fetch_dataframe(connection, sql_category, readonly=True)
        print(f"[COCKPIT RESULT] rows={0 if cat_df is None else len(cat_df)}", flush=True)

        sql_month = (
            f"SELECT DISTINCT SUBSTR(CAST(date_format(CAST({_quote_db_ident('Date')} AS DATE), 'yyyy-MM') AS STRING), 1, 7) AS month_key "
            f"FROM {invoice_fqn} WHERE {_quote_db_ident('Date')} IS NOT NULL ORDER BY 1 DESC"
        )
        print(f"[COCKPIT SQL] {sql_month}", flush=True)
        mo_df = fetch_dataframe(connection, sql_month, readonly=True)
        print(f"[COCKPIT RESULT] rows={0 if mo_df is None else len(mo_df)}", flush=True)

        if ch_df is not None and not ch_df.empty:
            filter_options["account_group"] = [str(v).strip() for v in ch_df.iloc[:, 0].tolist() if str(v).strip()]
        if br_df is not None and not br_df.empty:
            filter_options["brand"] = [str(v).strip() for v in br_df.iloc[:, 0].tolist() if str(v).strip()]
        if cat_df is not None and not cat_df.empty:
            filter_options["category"] = [str(v).strip() for v in cat_df.iloc[:, 0].tolist() if str(v).strip()]
        if mo_df is not None and not mo_df.empty:
            filter_options["months"] = [str(v).strip() for v in mo_df.iloc[:, 0].tolist() if str(v).strip()]
    except Exception:
        # Do not fail init if distinct option query fails; dashboard can still load.
        pass

    meta = {
        "source_mode": "databricks",
        "invoice_table": str(invoice_table),
        "target_table": str(target_table),
        "invoice_table_fqn": invoice_fqn,
        "target_table_fqn": target_fqn,
        "max_date": str(max_ts.date()),
        "curr_month_start": str(curr_start.date()),
        "month_key": str(max_ts.strftime("%Y-%m")),
        "working_days": int(working_days_val),
        "days_lapsed": int(days_lapsed_val),
        "days_left": int(days_left_val),
        "filter_options": filter_options,
    }
    return pd.DataFrame(), meta


def _sql_str_lit(value):
    return "'" + str(value or "").replace("'", "''") + "'"


def _run_db_query(connection, sql_text):
    print(f"[COCKPIT SQL] {sql_text}", flush=True)
    df = fetch_dataframe(connection, sql_text, readonly=True)
    print(f"[COCKPIT RESULT] rows={0 if df is None else len(df)}", flush=True)
    return df


def _db_scalar(connection, sql_text, default=0.0):
    df = _run_db_query(connection, sql_text)
    if df is None or df.empty:
        return default
    val = df.iloc[0, 0]
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    return val


def _db_metric_total(connection, table_fqn, metric_col, date_start, date_end, extra_conditions):
    date_expr = f"CAST({_quote_db_ident('Date')} AS DATE)"
    metric_expr = f"COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)"
    conditions = [
        f"{date_expr} >= DATE {_sql_str_lit(str(pd.Timestamp(date_start).date()))}",
        f"{date_expr} <= DATE {_sql_str_lit(str(pd.Timestamp(date_end).date()))}",
    ] + list(extra_conditions or [])
    sql_text = (
        f"SELECT SUM({metric_expr}) AS total "
        f"FROM {table_fqn} "
        f"WHERE {' AND '.join(conditions)}"
    )
    return float(_safe_float(_db_scalar(connection, sql_text, default=0.0)))


def _db_distinct_strings(connection, table_fqn, col_name):
    expr = _quote_db_ident(col_name)
    sql_text = (
        f"SELECT DISTINCT TRIM(CAST({expr} AS STRING)) AS v "
        f"FROM {table_fqn} "
        f"WHERE {expr} IS NOT NULL AND TRIM(CAST({expr} AS STRING)) <> '' "
        f"ORDER BY 1"
    )
    df = _run_db_query(connection, sql_text)
    if df is None or df.empty:
        return []
    out = []
    for v in df.iloc[:, 0].tolist():
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def _build_cockpit_filter_conditions(filters, table_kind="invoice"):
    f = filters if isinstance(filters, dict) else {}
    conditions = []

    channel_col = "Channel"
    brand_col = "Brand"
    category_col = "Category"

    account_group = str(f.get("account_group") or "All").strip()
    if table_kind == "invoice" and account_group and account_group.lower() != "all":
        conditions.append(
            f"TRIM(CAST({_quote_db_ident(channel_col)} AS STRING)) = {_sql_str_lit(account_group)}"
        )

    brand_val = str(f.get("brand") or "All").strip()
    if brand_val and brand_val.lower() != "all":
        conditions.append(
            f"TRIM(CAST({_quote_db_ident(brand_col)} AS STRING)) = {_sql_str_lit(brand_val)}"
        )

    bcra_val = str(f.get("bcra_dairy") or "All").strip().upper()
    cat_u = f"UPPER(TRIM(CAST({_quote_db_ident(category_col)} AS STRING)))"
    if bcra_val and bcra_val != "ALL":
        if bcra_val == "BCRA":
            conditions.append(f"{cat_u} NOT LIKE '%DAIRY%'")
        elif bcra_val == "DAIRY":
            conditions.append(f"{cat_u} LIKE '%DAIRY%'")
        elif bcra_val == "BOTH":
            pass
        elif bcra_val == "NONE":
            conditions.append(f"{cat_u} = 'NONE'")
        else:
            conditions.append(f"{cat_u} = {_sql_str_lit(bcra_val)}")

    return conditions


def _build_cockpit_payload_databricks(
    connection=None,
    invoice_table=DEFAULT_COCKPIT_INVOICE_TABLE,
    target_table=DEFAULT_COCKPIT_TARGET_TABLE,
    selected_month="",
    filters=None,
    filter_options_cache=None,
):
    invoice_table = str(invoice_table or DEFAULT_COCKPIT_INVOICE_TABLE).strip()
    target_table = str(target_table or DEFAULT_COCKPIT_TARGET_TABLE).strip()
    invoice_fqn = _quote_db_table_fqn(invoice_table)
    target_fqn = _quote_db_table_fqn(target_table)

    filters = filters if isinstance(filters, dict) else {}
    value_mode = str(filters.get("value_mode") or "value").strip().lower()
    if value_mode not in {"value", "volume"}:
        value_mode = "value"
    donut_mode = str(filters.get("donut_mode") or "business_channel").strip().lower()
    if donut_mode not in {"business_channel", "region"}:
        donut_mode = "business_channel"

    metric_col = "Volume" if value_mode == "volume" else "Value"
    target_metric_col = "Vol_Target" if value_mode == "volume" else "Values_Target"
    is_value_mode = (value_mode == "value")
    currency_scale = (1.0 / 10_000_000.0) if is_value_mode else 1.0

    own_connection = False
    if connection is None:
        try:
            connection = get_databricks_connection()
            own_connection = True
        except DatabricksConfigError as e:
            raise ValueError(str(e)) from e

    try:
        max_date_raw = _db_scalar(
            connection,
            f"SELECT MAX(CAST({_quote_db_ident('Date')} AS DATE)) AS max_date FROM {invoice_fqn}",
            default=None,
        )
        max_date_ts = pd.to_datetime(max_date_raw, errors="coerce")
        if pd.isna(max_date_ts):
            raise ValueError("Could not resolve max Date from invoice table.")
        max_date_ts = pd.Timestamp(max_date_ts).normalize()

        selected_raw = str(selected_month or "").strip()
        selected_match = re.match(r"^\d{4}-\d{2}$", selected_raw)
        if selected_match:
            selected_anchor = pd.to_datetime(selected_raw + "-01", errors="coerce")
            if pd.isna(selected_anchor):
                selected_anchor = max_date_ts.replace(day=1)
            else:
                selected_anchor = pd.Timestamp(selected_anchor).normalize()
        else:
            selected_anchor = max_date_ts.replace(day=1)

        curr_start = selected_anchor
        selected_key = selected_anchor.strftime("%Y-%m")
        max_key = max_date_ts.strftime("%Y-%m")
        month_last_day = monthrange(selected_anchor.year, selected_anchor.month)[1]
        full_month_end = selected_anchor.replace(day=month_last_day)
        curr_end = max_date_ts if selected_key == max_key else full_month_end

        working_days = int(monthrange(curr_start.year, curr_start.month)[1])
        days_lapsed = int(curr_end.day)
        days_left = max(0, working_days - days_lapsed)

        ly_start = (curr_start - pd.DateOffset(years=1)).replace(day=1)
        ly_month_last = monthrange(ly_start.year, ly_start.month)[1]
        ly_end_day = min(days_lapsed, ly_month_last)
        ly_end = ly_start.replace(day=ly_end_day)
        prev_end = curr_start - pd.Timedelta(days=1)
        prev_start = prev_end.replace(day=1)
        prev_end_aligned = min(prev_end, prev_start + pd.Timedelta(days=max(0, days_lapsed - 1)))
        month_key = curr_start.strftime("%Y-%m")

        invoice_filters = _build_cockpit_filter_conditions(filters, table_kind="invoice")
        target_filters = _build_cockpit_filter_conditions(filters, table_kind="target")

        mtd_raw = _db_metric_total(connection, invoice_fqn, metric_col, curr_start, curr_end, invoice_filters)
        lymtd_raw = _db_metric_total(connection, invoice_fqn, metric_col, ly_start, ly_end, invoice_filters)
        prev_mtd_raw = _db_metric_total(connection, invoice_fqn, metric_col, prev_start, prev_end_aligned, invoice_filters)
        mtd = float(mtd_raw) * currency_scale
        lymtd = float(lymtd_raw) * currency_scale
        prev_mtd = float(prev_mtd_raw) * currency_scale

        msp_conditions = [
            f"TRIM(CAST({_quote_db_ident('Monthly')} AS STRING)) = {_sql_str_lit(month_key)}"
        ] + target_filters
        msp_sql = (
            f"SELECT SUM(COALESCE(CAST({_quote_db_ident(target_metric_col)} AS DOUBLE), 0.0)) AS total "
            f"FROM {target_fqn} "
            f"WHERE {' AND '.join(msp_conditions)}"
        )
        msp_raw = float(_safe_float(_db_scalar(connection, msp_sql, default=0.0)))
        msp = msp_raw * currency_scale

        print(f"[COCKPIT VALUE] MTD raw={mtd_raw}, scaled={mtd}, mode={value_mode}", flush=True)
        print(f"[COCKPIT VALUE] LYMTD raw={lymtd_raw}, scaled={lymtd}, mode={value_mode}", flush=True)
        print(f"[COCKPIT VALUE] SEQ raw={prev_mtd_raw}, scaled={prev_mtd}, mode={value_mode}", flush=True)
        print(f"[COCKPIT VALUE] MSP raw={msp_raw}, scaled={msp}, mode={value_mode}", flush=True)

        sly_pct = _pct_delta(mtd, lymtd)
        seq_pct = _pct_delta(mtd, prev_mtd)
        le = (mtd / max(1, days_lapsed)) * working_days
        # ECO SLY/SEQ baselines are straight-line closing estimates in Crs.
        eco_sly_base = (lymtd / max(1, days_lapsed)) * working_days
        eco_seq_base = (prev_mtd / max(1, days_lapsed)) * working_days
        rrr = max(0.0, (msp - mtd) / max(1, days_left))
        drr = mtd / max(1, days_lapsed)

        daily_conditions = [
            f"CAST({_quote_db_ident('Date')} AS DATE) >= DATE {_sql_str_lit(str(curr_start.date()))}",
            f"CAST({_quote_db_ident('Date')} AS DATE) <= DATE {_sql_str_lit(str(curr_end.date()))}",
        ] + invoice_filters
        daily_sql = (
            f"SELECT CAST({_quote_db_ident('Date')} AS DATE) AS d, "
            f"SUM(COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)) AS v "
            f"FROM {invoice_fqn} WHERE {' AND '.join(daily_conditions)} "
            f"GROUP BY 1 ORDER BY 1 DESC LIMIT 3"
        )
        daily_df = _run_db_query(connection, daily_sql)
        day_map = {}
        if daily_df is not None and not daily_df.empty:
            for _, row in daily_df.iterrows():
                d = pd.to_datetime(row.iloc[0], errors="coerce")
                if pd.isna(d):
                    continue
                day_map[d.normalize()] = float(_safe_float(row.iloc[1])) * currency_scale
        d1_date = curr_end - pd.Timedelta(days=2)
        d2_date = curr_end - pd.Timedelta(days=1)
        d3_date = curr_end
        d1 = float(day_map.get(d1_date.normalize(), 0.0))
        d2 = float(day_map.get(d2_date.normalize(), 0.0))
        d3 = float(day_map.get(d3_date.normalize(), 0.0))
        l3 = float((d1 + d2 + d3) / 3.0)
        if daily_df is not None and not daily_df.empty:
            daily_sample_raw = float(_safe_float(daily_df.iloc[0, 1]))
            print(
                f"[COCKPIT VALUE] daily sample raw={daily_sample_raw}, scaled={daily_sample_raw * currency_scale}, mode={value_mode}",
                flush=True,
            )

        # ECO in Crs: momentum-based estimate using recent pace.
        eco = mtd + (l3 * max(0, days_left))
        # ECO comparisons shown as percentages.
        eco_sly = _pct_delta(eco, eco_sly_base)
        eco_seq = _pct_delta(eco, eco_seq_base)

        def _contrib_for_dim(dim_col):
            dim_ident = _quote_db_ident(dim_col)
            mtd_where = [
                f"CAST({_quote_db_ident('Date')} AS DATE) >= DATE {_sql_str_lit(str(curr_start.date()))}",
                f"CAST({_quote_db_ident('Date')} AS DATE) <= DATE {_sql_str_lit(str(curr_end.date()))}",
            ] + invoice_filters
            ly_where = [
                f"CAST({_quote_db_ident('Date')} AS DATE) >= DATE {_sql_str_lit(str(pd.Timestamp(ly_start).date()))}",
                f"CAST({_quote_db_ident('Date')} AS DATE) <= DATE {_sql_str_lit(str(pd.Timestamp(ly_end).date()))}",
            ] + invoice_filters
            mtd_sql = (
                f"SELECT CAST({dim_ident} AS STRING) AS k, "
                f"SUM(COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)) AS v "
                f"FROM {invoice_fqn} WHERE {' AND '.join(mtd_where)} "
                f"GROUP BY 1"
            )
            ly_sql = (
                f"SELECT CAST({dim_ident} AS STRING) AS k, "
                f"SUM(COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)) AS v "
                f"FROM {invoice_fqn} WHERE {' AND '.join(ly_where)} "
                f"GROUP BY 1"
            )
            mdf = _run_db_query(connection, mtd_sql)
            ldf = _run_db_query(connection, ly_sql)
            cm = {}
            lm = {}
            contrib_sample_raw = None
            if mdf is not None and not mdf.empty:
                for _, row in mdf.iterrows():
                    key = str(row.iloc[0] or "").strip() or "Unknown"
                    raw_v = float(_safe_float(row.iloc[1]))
                    if contrib_sample_raw is None:
                        contrib_sample_raw = raw_v
                    cm[key] = raw_v * currency_scale
            if ldf is not None and not ldf.empty:
                for _, row in ldf.iterrows():
                    key = str(row.iloc[0] or "").strip() or "Unknown"
                    lm[key] = float(_safe_float(row.iloc[1])) * currency_scale
            if contrib_sample_raw is not None:
                print(
                    f"[COCKPIT VALUE] contrib dim={dim_col} sample raw={contrib_sample_raw}, scaled={contrib_sample_raw * currency_scale}, mode={value_mode}",
                    flush=True,
                )
            keys = set(cm.keys()) | set(lm.keys())
            rows = [{"k": k, "cmtd": cm.get(k, 0.0), "lymtd": lm.get(k, 0.0)} for k in keys]
            rows.sort(key=lambda x: x["cmtd"], reverse=True)
            rows = rows[:8]
            return {
                "labels": [str(r["k"]) for r in rows],
                "cmtd": [float(r["cmtd"]) for r in rows],
                "lymtd": [float(r["lymtd"]) for r in rows],
            }

        business_contrib = _contrib_for_dim("Channel")
        account_contrib = _contrib_for_dim("Brand")
        category_contrib = _contrib_for_dim("Category")

        map_where = [
            f"CAST({_quote_db_ident('Date')} AS DATE) >= DATE {_sql_str_lit(str(curr_start.date()))}",
            f"CAST({_quote_db_ident('Date')} AS DATE) <= DATE {_sql_str_lit(str(curr_end.date()))}",
        ] + invoice_filters
        map_sql = (
            f"SELECT CAST({_quote_db_ident('Region')} AS STRING) AS region, "
            f"SUM(COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)) AS val "
            f"FROM {invoice_fqn} WHERE {' AND '.join(map_where)} "
            f"GROUP BY 1 ORDER BY 2 DESC"
        )
        map_df = _run_db_query(connection, map_sql)
        map_rows = []
        map_sample_raw = None
        if map_df is not None and not map_df.empty:
            for _, row in map_df.iterrows():
                rk = str(row.iloc[0] or "").strip()
                if not rk:
                    continue
                raw_val = float(_safe_float(row.iloc[1]))
                if map_sample_raw is None:
                    map_sample_raw = raw_val
                map_rows.append((rk, raw_val * currency_scale))
        if map_sample_raw is not None:
            print(
                f"[COCKPIT VALUE] map sample raw={map_sample_raw}, scaled={map_sample_raw * currency_scale}, mode={value_mode}",
                flush=True,
            )
        map_total = float(sum(v for _, v in map_rows))
        map_avg = float((sum(v for _, v in map_rows) / len(map_rows))) if map_rows else 0.0
        map_points = []
        for idx, (region, val) in enumerate(map_rows):
            lat, lon = _region_coords(region, idx)
            pct = (val / map_total * 100.0) if map_total > 0 else 0.0
            map_points.append(
                {
                    "region": str(region),
                    "lat": float(lat),
                    "lon": float(lon),
                    "value": float(val),
                    "pct": round(float(pct), 2),
                    "score": float(val - map_avg),
                }
            )

        donut_dim_col = "Channel" if donut_mode == "business_channel" else "Region"
        donut_dim_ident = _quote_db_ident(donut_dim_col)
        donut_sql = (
            f"SELECT CAST({donut_dim_ident} AS STRING) AS k, "
            f"SUM(COALESCE(CAST({_quote_db_ident(metric_col)} AS DOUBLE), 0.0)) AS v "
            f"FROM {invoice_fqn} WHERE {' AND '.join(map_where)} "
            f"GROUP BY 1 ORDER BY 2 DESC LIMIT 8"
        )
        donut_df = _run_db_query(connection, donut_sql)
        donut_labels = []
        donut_values = []
        donut_sample_raw = None
        if donut_df is not None and not donut_df.empty:
            for _, row in donut_df.iterrows():
                donut_labels.append(str(row.iloc[0] or "Unknown"))
                raw_val = float(_safe_float(row.iloc[1]))
                if donut_sample_raw is None:
                    donut_sample_raw = raw_val
                donut_values.append(raw_val * currency_scale)
        if donut_sample_raw is not None:
            print(
                f"[COCKPIT VALUE] donut sample raw={donut_sample_raw}, scaled={donut_sample_raw * currency_scale}, mode={value_mode}",
                flush=True,
            )
        donut_total = float(sum(donut_values))
        donut = {
            "labels": donut_labels,
            "values": donut_values,
            "percents": [round((v / donut_total * 100.0) if donut_total > 0 else 0.0, 2) for v in donut_values],
            "mode": donut_mode,
        }

        retailer_expr = f"TRIM(CAST({_quote_db_ident('Retailer')} AS STRING))"
        valid_retailer_conditions = [
            f"{_quote_db_ident('Retailer')} IS NOT NULL",
            f"{retailer_expr} NOT IN ('', 'None', 'none', 'NULL', 'null')",
        ]
        outlet_where = map_where + valid_retailer_conditions
        outlet_sql = (
            f"SELECT COUNT(DISTINCT {retailer_expr}) AS cnt "
            f"FROM {invoice_fqn} WHERE {' AND '.join(outlet_where)}"
        )
        visited = int(_safe_float(_db_scalar(connection, outlet_sql, default=0)))
        print(f"[COCKPIT VALUE] outlet raw count={visited}", flush=True)

        # Fallback to Distributor if Retailer is sparsely populated.
        if visited <= 0:
            dist_expr = f"TRIM(CAST({_quote_db_ident('Distributor')} AS STRING))"
            outlet_fallback_where = map_where + [
                f"{_quote_db_ident('Distributor')} IS NOT NULL",
                f"{dist_expr} NOT IN ('', 'None', 'none', 'NULL', 'null')",
            ]
            outlet_fallback_sql = (
                f"SELECT COUNT(DISTINCT {dist_expr}) AS cnt "
                f"FROM {invoice_fqn} WHERE {' AND '.join(outlet_fallback_where)}"
            )
            visited = int(_safe_float(_db_scalar(connection, outlet_fallback_sql, default=0)))
            print(f"[COCKPIT VALUE] outlet fallback distributor count={visited}", flush=True)

        # Source does not contain an explicit productivity flag.
        productive = int(max(0, visited))
        non_productive = 0

        cached = filter_options_cache if isinstance(filter_options_cache, dict) else {}
        account_options = list(cached.get("account_group") or [])
        brand_options = list(cached.get("brand") or [])
        category_options = list(cached.get("category") or [])
        month_options = list(cached.get("months") or [])
        if not account_options:
            account_options = _db_distinct_strings(connection, invoice_fqn, "Channel")
        if not brand_options:
            brand_options = _db_distinct_strings(connection, invoice_fqn, "Brand")
        if not category_options:
            category_options = _db_distinct_strings(connection, invoice_fqn, "Category")
        if not month_options:
            month_sql = (
                f"SELECT DISTINCT SUBSTR(CAST(date_format(CAST({_quote_db_ident('Date')} AS DATE), 'yyyy-MM') AS STRING), 1, 7) AS month_key "
                f"FROM {invoice_fqn} WHERE {_quote_db_ident('Date')} IS NOT NULL ORDER BY 1 DESC"
            )
            month_df = _run_db_query(connection, month_sql)
            if month_df is not None and not month_df.empty:
                month_options = [str(v).strip() for v in month_df.iloc[:, 0].tolist() if str(v).strip()]
        has_dairy = any("DAIRY" in str(c).upper() for c in category_options)
        has_non_dairy = any("DAIRY" not in str(c).upper() for c in category_options)
        bcra_options = ["All"]
        if has_non_dairy:
            bcra_options.append("BCRA")
        if has_dairy:
            bcra_options.append("DAIRY")
        if has_non_dairy and has_dairy:
            bcra_options.append("BOTH")

        charts = {
            "business_channel_contribution": business_contrib,
            "account_group_contribution": account_contrib,
            "category_contribution": category_contrib,
            "daily_bars": {
                "labels": ["D 1", "D 2", "D 3", "L3DAvg"],
                "values": [float(d1), float(d2), float(d3), float(l3)],
                "rrr": float(rrr),
            },
            "map_points": map_points,
            "donut": donut,
        }

        date_label = pd.to_datetime(curr_end).strftime("%d-%b").upper()
        payload = {
            "header": {
                "date": date_label,
                "working_days": int(working_days),
                "days_lapsed": int(days_lapsed),
                "days_left": int(days_left),
                "value_mode_label": "Value In Cr's" if value_mode == "value" else "Volume",
            },
            "filters": {
                "options": {
                    "account_group": ["All"] + [x for x in account_options if str(x).strip()],
                    "bcra_dairy": bcra_options,
                    "brand": ["All"] + [x for x in brand_options if str(x).strip()],
                    "months": [x for x in month_options if str(x).strip()],
                },
                "selected": {
                    "account_group": str(filters.get("account_group") or "All"),
                    "bcra_dairy": str(filters.get("bcra_dairy") or "All"),
                    "brand": str(filters.get("brand") or "All"),
                    "value_mode": value_mode,
                    "donut_mode": donut_mode,
                    "selected_month": month_key,
                },
            },
            "kpis": {
                "msp": {"value": float(msp), "display": _format_value(msp)},
                "le": {"value": float(le), "display": _format_value(le)},
                "mtd": {"value": float(mtd), "display": _format_value(mtd)},
                "sly_pct": round(float(sly_pct), 2),
                "seq_pct": round(float(seq_pct), 2),
                "eco": {"value": float(eco), "display": f"{float(eco):,.1f}"},
                "eco_sly": round(float(eco_sly), 2),
                "eco_sly_display": f"{float(eco_sly):.2f}%",
                "eco_seq": round(float(eco_seq), 2),
                "eco_seq_display": f"{float(eco_seq):.2f}%",
                "rrr": round(float(rrr), 2),
                "drr": round(float(drr), 2),
                "productive_outlets": int(productive),
                "non_productive_outlets": int(non_productive),
                "visited_outlets": int(visited),
            },
            "charts": charts,
        }
        return payload
    finally:
        if own_connection:
            try:
                connection.close()
            except Exception:
                pass


def serialize_df_for_session(df):
    return df.to_json(orient="split", date_format="iso")


def deserialize_df_from_session(payload):
    if not payload:
        return None
    try:
        parsed = pd.read_json(io.StringIO(payload), orient="split")
        return _normalize_cockpit_df(parsed)
    except Exception:
        return None


def _apply_filters(df, filters):
    out = df.copy()
    f = filters if isinstance(filters, dict) else {}

    acct_val = str(f.get("account_group") or "All").strip()
    if acct_val and acct_val.lower() != "all":
        out = out[out["_account_group"].astype(str).str.strip() == acct_val]

    # BCRA & Dairy inclusive semantics by selected value.
    bcra_val = str(f.get("bcra_dairy") or "All").strip().upper()
    if bcra_val and bcra_val != "ALL":
        bcra_col = out["_bcra_dairy"].astype(str).str.upper().str.strip()
        if bcra_val == "BCRA":
            out = out[bcra_col.isin(["BCRA", "BOTH"])]
        elif bcra_val == "DAIRY":
            out = out[bcra_col.isin(["DAIRY", "BOTH"])]
        elif bcra_val == "BOTH":
            out = out[bcra_col != "NONE"]
        elif bcra_val == "NONE":
            out = out[bcra_col == "NONE"]
        else:
            out = out[bcra_col == bcra_val]

    brand_val = str(f.get("brand") or "All").strip()
    if brand_val and brand_val.lower() != "all":
        out = out[out["_brand"].astype(str).str.strip() == brand_val]

    return out


def _month_windows(max_date):
    curr_start = max_date.replace(day=1)
    curr_end = max_date
    working_days = monthrange(max_date.year, max_date.month)[1]
    days_lapsed = int(max_date.day)
    days_left = max(working_days - days_lapsed, 0)

    ly_end = max_date - pd.DateOffset(years=1)
    ly_start = ly_end.replace(day=1)

    prev_month_end = curr_start - pd.Timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    prev_month_aligned_end = min(
        prev_month_end,
        prev_month_start + pd.Timedelta(days=max(0, days_lapsed - 1)),
    )
    return {
        "curr_start": curr_start,
        "curr_end": curr_end,
        "ly_start": ly_start,
        "ly_end": ly_end,
        "prev_start": prev_month_start,
        "prev_end": prev_month_aligned_end,
        "working_days": int(working_days),
        "days_lapsed": int(days_lapsed),
        "days_left": int(days_left),
    }


REGION_COORDS = {
    "north1": (31.10, 76.40),
    "north2": (28.60, 77.20),
    "west": (19.25, 73.20),
    "east": (22.60, 88.30),
    "central": (23.30, 79.90),
    "south1": (12.90, 79.10),
    "south2": (10.20, 76.30),
    "cent": (23.30, 79.90),
    "eca1": (22.60, 88.30),
    "nde1": (31.10, 76.40),
    "nde2": (28.60, 77.20),
    "sch2": (12.90, 79.10),
    "sch3": (10.20, 76.30),
    "wmu1": (19.25, 73.20),
}


def _region_coords(name, idx):
    key = _norm(name)
    if key in REGION_COORDS:
        return REGION_COORDS[key]
    fallback = [
        (28.6, 77.2),
        (26.9, 80.9),
        (23.0, 72.6),
        (22.5, 88.4),
        (19.1, 72.9),
        (13.0, 80.3),
        (17.4, 78.5),
        (11.0, 76.9),
    ]
    return fallback[idx % len(fallback)]


def _format_value(v):
    n = _safe_float(v)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n:,.0f}"
    return f"{n:.2f}"


def _smart_kpi_agg(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0
    nz = s[s.abs() > 1e-12]
    if nz.empty:
        return 0.0
    # Snapshot-style KPI columns are often repeated per row; in that case use max.
    unique_n = int(nz.nunique(dropna=True))
    if unique_n <= 3 or (unique_n / max(1, len(nz))) <= 0.02:
        return float(nz.max())
    return float(nz.sum())


def _contrib_table(curr_df, ly_df, dim_col, metric_col, top_n=8):
    curr = curr_df.groupby(dim_col)[metric_col].sum() if not curr_df.empty else pd.Series(dtype=float)
    ly = ly_df.groupby(dim_col)[metric_col].sum() if not ly_df.empty else pd.Series(dtype=float)
    all_idx = curr.index.union(ly.index)
    if len(all_idx) == 0:
        return {"labels": [], "cmtd": [], "lymtd": []}
    table = pd.DataFrame(
        {
            "cmtd": curr.reindex(all_idx).fillna(0.0),
            "lymtd": ly.reindex(all_idx).fillna(0.0),
        }
    )
    table = table.sort_values("cmtd", ascending=False)
    if top_n is not None and int(top_n) > 0:
        table = table.head(int(top_n))
    return {
        "labels": [str(x) for x in table.index],
        "cmtd": [float(v) for v in table["cmtd"].tolist()],
        "lymtd": [float(v) for v in table["lymtd"].tolist()],
    }


def _contrib_table_dual(source_df, dim_col, cmtd_col, lymtd_col, top_n=8):
    if source_df is None or source_df.empty:
        return {"labels": [], "cmtd": [], "lymtd": []}
    cmtd_s = pd.to_numeric(source_df[cmtd_col], errors="coerce").fillna(0.0)
    lymtd_s = pd.to_numeric(source_df[lymtd_col], errors="coerce").fillna(0.0)
    temp = pd.DataFrame({dim_col: source_df[dim_col].astype(str), "_cmtd": cmtd_s, "_lymtd": lymtd_s})
    table = temp.groupby(dim_col, as_index=True).agg(cmtd=("_cmtd", "sum"), lymtd=("_lymtd", "sum"))
    if table.empty:
        return {"labels": [], "cmtd": [], "lymtd": []}

    total_cmtd = float(table["cmtd"].sum())
    total_lymtd = float(table["lymtd"].sum())
    if total_cmtd > 0.0 and total_lymtd > 0.0:
        sly_input = source_df.get("_sly_pct_col", pd.Series(dtype=float))
        avg_sly = float(pd.to_numeric(sly_input, errors="coerce").fillna(0.0).mean())
        denom = 1.0 + (avg_sly / 100.0)
        scale = (1.0 / denom) if abs(denom) > 1e-9 else 1.0
        table["lymtd"] = table["cmtd"] * scale

    table = table.sort_values("cmtd", ascending=False)
    if top_n is not None and int(top_n) > 0:
        table = table.head(int(top_n))
    return {
        "labels": [str(x) for x in table.index],
        "cmtd": [float(v) for v in table["cmtd"].tolist()],
        "lymtd": [float(v) for v in table["lymtd"].tolist()],
    }


def build_cockpit_payload(
    df=None,
    filters=None,
    data_source="excel",
    databricks_config=None,
    connection=None,
    invoice_table=None,
    target_table=None,
    selected_month=None,
):
    source = str(data_source or "excel").strip().lower()
    if source == "databricks":
        cfg = databricks_config if isinstance(databricks_config, dict) else {}
        inv = str(invoice_table or cfg.get("invoice_table") or DEFAULT_COCKPIT_INVOICE_TABLE).strip()
        tgt = str(target_table or cfg.get("target_table") or DEFAULT_COCKPIT_TARGET_TABLE).strip()
        sel_month = str(selected_month or (filters or {}).get("selected_month") or "").strip()
        return _build_cockpit_payload_databricks(
            connection=connection,
            invoice_table=inv,
            target_table=tgt,
            selected_month=sel_month,
            filters=filters,
            filter_options_cache=cfg.get("filter_options"),
        )

    if df is None or df.empty:
        return {
            "error": "No data loaded",
            "filters": {},
            "kpis": {},
            "charts": {},
        }
    df = _normalize_cockpit_df(df)

    filters = filters if isinstance(filters, dict) else {}
    value_mode = str(filters.get("value_mode") or "value").strip().lower()
    if value_mode not in {"value", "volume"}:
        value_mode = "value"
    donut_mode = str(filters.get("donut_mode") or "business_channel").strip().lower()
    if donut_mode not in {"business_channel", "region"}:
        donut_mode = "business_channel"

    work_df = _apply_filters(df, filters)
    metric_col = "_volume_metric" if value_mode == "volume" else "_value_metric"
    ly_metric_col = "_ly_volume_metric" if value_mode == "volume" else "_ly_value_metric"

    if work_df.empty:
        work_df = df.copy()

    # Fallback: if LY volume metric is missing/zero, use LY value metric.
    if ly_metric_col == "_ly_volume_metric":
        ly_col_data = pd.to_numeric(
            work_df.get("_ly_volume_metric", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        if ly_col_data.abs().sum() < 1e-9:
            ly_metric_col = "_ly_value_metric"

    if "_date" in work_df.columns:
        work_df = work_df.copy()
        work_df["_date"] = pd.to_datetime(work_df["_date"], errors="coerce")
    has_date = "_date" in work_df.columns and work_df["_date"].notna().any()
    if has_date:
        max_date = work_df["_date"].max()
        windows = _month_windows(max_date)
        date_series = work_df["_date"]
        curr_mask = (date_series >= windows["curr_start"]) & (date_series <= windows["curr_end"])
        ly_mask = (date_series >= windows["ly_start"]) & (date_series <= windows["ly_end"])
        prev_mask = (date_series >= windows["prev_start"]) & (date_series <= windows["prev_end"])
        curr_df = work_df[curr_mask]
        ly_df = work_df[ly_mask]
        prev_df = work_df[prev_mask]
    else:
        max_date = datetime.utcnow()
        windows = {
            "working_days": 30,
            "days_lapsed": 15,
            "days_left": 15,
        }
        curr_df = work_df
        ly_df = work_df.iloc[0:0]
        prev_df = work_df.iloc[0:0]

    has_explicit_ly = (
        ly_metric_col in work_df.columns
        and pd.to_numeric(work_df[ly_metric_col], errors="coerce").fillna(0.0).abs().sum() > 0
    )
    if ly_df.empty and not prev_df.empty and not has_explicit_ly:
        ly_df = prev_df

    # Left-panel KPIs use explicit cockpit columns when present.
    def _sum_col(frame, col_name):
        if col_name not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[col_name], errors="coerce").fillna(0.0).sum())

    def _series_or_empty(frame, col_name):
        if col_name not in frame.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[col_name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _max_positive(series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0]
        return int(s.max()) if not s.empty else None

    def _min_positive(series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0]
        return int(s.min()) if not s.empty else None

    wd_from_col = _max_positive(work_df.get("_working_days_col", pd.Series(dtype=float)))
    dl_from_col = _max_positive(work_df.get("_days_lapsed_col", pd.Series(dtype=float)))
    dr_from_col = _min_positive(work_df.get("_days_left_col", pd.Series(dtype=float)))

    days_lapsed = max(1, int(dl_from_col if dl_from_col is not None else windows.get("days_lapsed", 15)))
    working_days = max(days_lapsed, int(wd_from_col if wd_from_col is not None else windows.get("working_days", 30)))
    days_left = max(0, int(dr_from_col if dr_from_col is not None else windows.get("days_left", working_days - days_lapsed)))

    # KPI formulas per cockpit spec (daily-aggregated; avoid row-level inflation).
    date_norm = (
        pd.to_datetime(work_df["_date"], errors="coerce").dt.normalize()
        if "_date" in work_df.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    daily = pd.DataFrame(columns=["_day", "_mtd_metric", "_ly_value_metric", "_eco_thousands", "_msp_metric"])
    def _num_col(col_name):
        if col_name in work_df.columns:
            return pd.to_numeric(work_df[col_name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=work_df.index, dtype=float)

    if has_date:
        daily = pd.DataFrame(
            {
                "_day": date_norm,
                "_mtd_metric": _num_col("_mtd_metric"),
                "_ly_value_metric": _num_col("_ly_value_metric"),
                "_eco_thousands": _num_col("_eco_thousands"),
                "_msp_metric": _num_col("_msp_metric"),
            }
        ).dropna(subset=["_day"])
        if not daily.empty:
            daily = (
                daily.groupby("_day", as_index=False)[["_mtd_metric", "_ly_value_metric", "_eco_thousands", "_msp_metric"]]
                .sum()
                .sort_values("_day")
            )

    eco_series = pd.to_numeric(work_df.get("_eco_thousands", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    # MTD must come from current-month scope, not just the last day bucket.
    if not curr_df.empty and "_mtd_metric" in curr_df.columns:
        mtd = round(float(_smart_kpi_agg(curr_df["_mtd_metric"])), 2)
    else:
        mtd = round(float(_smart_kpi_agg(work_df["_mtd_metric"]) if "_mtd_metric" in work_df.columns else 0.0), 2)

    # MSP: current-month only, dedup join fanout by grouped keys and
    # aggregate with max-per-group.
    msp_scope_df = curr_df if not curr_df.empty else work_df
    msp = 0.0
    if not msp_scope_df.empty and "_msp_metric" in msp_scope_df.columns:
        msp_tmp = msp_scope_df.copy()
        if has_date and "_date" in msp_tmp.columns:
            msp_tmp["_msp_month_key"] = pd.to_datetime(msp_tmp["_date"], errors="coerce").dt.strftime("%Y-%m")
            msp_tmp["_msp_month_key"] = msp_tmp["_msp_month_key"].fillna("")
        else:
            msp_tmp["_msp_month_key"] = ""

        for col in ["_region", "_business_channel", "_brand", "_category"]:
            if col not in msp_tmp.columns:
                msp_tmp[col] = ""
            else:
                msp_tmp[col] = msp_tmp[col].astype(str).replace({"nan": "", "None": ""}).fillna("").str.strip()

        msp_tmp["_msp_metric"] = pd.to_numeric(msp_tmp["_msp_metric"], errors="coerce").fillna(0.0)
        group_cols = ["_msp_month_key", "_region", "_business_channel", "_brand", "_category"]
        msp = round(float(msp_tmp.groupby(group_cols, dropna=False)["_msp_metric"].max().sum()), 2)
    # Requested formula: ECO row-level snapshot mean (in thousands).
    eco = round(float(eco_series.mean()) if not eco_series.empty else 0.0, 2)
    le = round((mtd / max(1, days_lapsed)) * working_days, 2)

    lymtd = _sum_col(work_df, "_ly_value_metric")
    prev_mtd = float(prev_df["_mtd_metric"].sum()) if (not prev_df.empty and "_mtd_metric" in prev_df.columns) else 0.0

    eco_sly_base = (lymtd / max(1, days_lapsed)) * working_days
    eco_seq_base = (prev_mtd / max(1, days_lapsed)) * working_days

    sly_series = _series_or_empty(work_df, "_sly_pct_col")
    seq_series = _series_or_empty(work_df, "_seq_pct_col")
    # When LY/previous baseline is unavailable in this dataset window,
    # _pct_delta naturally returns 0.00%; keep numeric tiles stable.
    sly_pct = float(sly_series.dropna().mean()) if sly_series.dropna().shape[0] else _pct_delta(mtd, lymtd)
    seq_pct = float(seq_series.dropna().mean()) if seq_series.dropna().shape[0] else _pct_delta(mtd, prev_mtd)
    eco_sly = round(_pct_delta(eco, eco_sly_base), 2)
    eco_seq = round(_pct_delta(eco, eco_seq_base), 2)

    rrr = round(max(0.0, (msp - mtd) / max(1, days_left)), 2)
    drr = round(mtd / max(1, days_lapsed), 2)

    # Section 6: explicit D1/D2/D3 from max-date -2/-1/0 and L3DAvg.
    if has_date:
        day_df = work_df.copy()
        day_df["_day"] = pd.to_datetime(day_df["_date"], errors="coerce").dt.normalize()
        day_metric_col = "_daily_value_metric" if value_mode == "value" else "_volume_metric"
        if day_metric_col not in day_df.columns:
            day_metric_col = "_value_metric" if value_mode == "value" else "_volume_metric"
        daily = day_df.groupby("_day")[day_metric_col].sum()
        max_day = day_df["_day"].max()
        d1_date = max_day - pd.Timedelta(days=2)
        d2_date = max_day - pd.Timedelta(days=1)
        d3_date = max_day
        d1 = float(daily.get(d1_date, 0.0))
        d2 = float(daily.get(d2_date, 0.0))
        d3 = float(daily.get(d3_date, 0.0))
        l3 = float((d1 + d2 + d3) / 3.0)
    else:
        base = mtd / max(days_lapsed, 1)
        d1, d2, d3 = base * 0.96, base * 1.01, base * 0.99
        l3 = float((d1 + d2 + d3) / 3.0)

    outlet_flags = (
        work_df[["_outlet", "_productive_flag"]]
        .drop_duplicates("_outlet")
        .groupby("_productive_flag")["_outlet"]
        .nunique()
    )
    productive = int(outlet_flags.get("Productive", 0))
    non_productive = int(outlet_flags.get("Non Productive", 0))
    visited = int(productive + non_productive)

    # Map / region contribution
    map_source = curr_df if not curr_df.empty else work_df
    region_table = map_source.groupby("_region")[metric_col].sum().sort_values(ascending=False)
    region_total = float(region_table.sum()) if len(region_table) else 0.0
    avg_val = float(region_table.mean()) if len(region_table) else 0.0
    map_points = []
    for i, (region, val) in enumerate(region_table.items()):
        lat, lon = _region_coords(region, i)
        pct = (float(val) / region_total * 100.0) if region_total > 0 else 0.0
        score = float(val - avg_val)
        map_points.append(
            {
                "region": str(region),
                "lat": float(lat),
                "lon": float(lon),
                "value": float(val),
                "pct": round(pct, 2),
                "score": score,
            }
        )

    # Donut
    donut_dim = "_business_channel" if donut_mode == "business_channel" else "_region"
    donut_table = map_source.groupby(donut_dim)[metric_col].sum().sort_values(ascending=False).head(8)
    donut_total = float(donut_table.sum()) if len(donut_table) else 0.0
    donut = {
        "labels": [str(x) for x in donut_table.index],
        "values": [float(v) for v in donut_table.tolist()],
        "percents": [round((float(v) / donut_total * 100.0) if donut_total > 0 else 0.0, 2) for v in donut_table.tolist()],
        "mode": donut_mode,
    }

    # Re-apply active filters to period slices used by right-panel contribution charts.
    contrib_curr = _apply_filters(curr_df, filters) if not curr_df.empty else work_df
    if contrib_curr.empty:
        contrib_curr = _apply_filters(work_df, filters)
    if contrib_curr.empty:
        contrib_curr = work_df
    contrib_ly = _apply_filters(ly_df, filters) if not ly_df.empty else pd.DataFrame()

    def _safe_ly_col(df_in, preferred_col, fallback_col):
        data = pd.to_numeric(df_in.get(preferred_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if data.abs().sum() < 1e-9:
            return fallback_col
        return preferred_col

    ly_metric_col = _safe_ly_col(contrib_curr, ly_metric_col, "_ly_value_metric")

    if has_explicit_ly:
        business_contrib = _contrib_table_dual(contrib_curr, "_business_channel", metric_col, ly_metric_col, top_n=8)
        account_contrib = _contrib_table_dual(contrib_curr, "_account_group", metric_col, ly_metric_col, top_n=8)
        category_contrib = _contrib_table_dual(contrib_curr, "_category", metric_col, ly_metric_col, top_n=8)
    else:
        business_contrib = _contrib_table(contrib_curr, contrib_ly, "_business_channel", metric_col, top_n=8)
        account_contrib = _contrib_table(contrib_curr, contrib_ly, "_account_group", metric_col, top_n=8)
        category_contrib = _contrib_table(contrib_curr, contrib_ly, "_category", metric_col, top_n=8)

    charts = {
        "business_channel_contribution": business_contrib,
        "account_group_contribution": account_contrib,
        "category_contribution": category_contrib,
        "daily_bars": {
            "labels": ["D 1", "D 2", "D 3", "L3DAvg"],
            "values": [float(d1), float(d2), float(d3), float(l3)],
            "rrr": float(rrr),
        },
        "map_points": map_points,
        "donut": donut,
    }

    all_filters = {
        "account_group": ["All"] + sorted([str(x) for x in df["_account_group"].dropna().astype(str).unique().tolist()]),
        "bcra_dairy": ["All"] + sorted([str(x) for x in df["_bcra_dairy"].dropna().astype(str).unique().tolist()]),
        "brand": ["All"] + sorted([str(x) for x in df["_brand"].dropna().astype(str).unique().tolist()]),
    }

    date_label = pd.to_datetime(max_date).strftime("%d-%b").upper() if has_date else datetime.utcnow().strftime("%d-%b").upper()

    return {
        "header": {
            "date": date_label,
            "working_days": int(working_days),
            "days_lapsed": int(days_lapsed),
            "days_left": int(days_left),
            "value_mode_label": "Value In Cr's" if value_mode == "value" else "Volume",
        },
        "filters": {
            "options": all_filters,
            "selected": {
                "account_group": str(filters.get("account_group") or "All"),
                "bcra_dairy": str(filters.get("bcra_dairy") or "All"),
                "brand": str(filters.get("brand") or "All"),
                "value_mode": value_mode,
                "donut_mode": donut_mode,
            },
        },
        "kpis": {
            "msp": {"value": float(msp), "display": _format_value(msp)},
            "le": {"value": float(le), "display": _format_value(le)},
            "mtd": {"value": float(mtd), "display": _format_value(mtd)},
            "sly_pct": round(float(sly_pct), 2),
            "seq_pct": round(float(seq_pct), 2),
            "eco": {"value": float(eco), "display": f"{float(eco):,.1f}"},
            "eco_sly": round(float(eco_sly), 2),
            "eco_sly_display": f"{float(eco_sly):.2f}%",
            "eco_seq": round(float(eco_seq), 2),
            "eco_seq_display": f"{float(eco_seq):.2f}%",
            "rrr": round(float(rrr), 2),
            "drr": round(float(drr), 2),
            "productive_outlets": int(productive),
            "non_productive_outlets": int(non_productive),
            "visited_outlets": int(visited),
        },
        "charts": charts,
    }


def build_cockpit_ai_reply(user_prompt, cockpit_payload):
    prompt = str(user_prompt or "").strip()
    if not prompt:
        return "Ask me about KPI trends, region performance, category contribution, or outlet productivity."

    p = prompt.lower()
    k = (cockpit_payload or {}).get("kpis", {})
    c = (cockpit_payload or {}).get("charts", {})

    if any(x in p for x in ["mtd", "month to date", "sales"]):
        return f"Current MTD is {k.get('mtd', {}).get('display', 'NA')} and LE is {k.get('le', {}).get('display', 'NA')}."
    if any(x in p for x in ["region", "map"]):
        points = c.get("map_points", []) or []
        if points:
            top = max(points, key=lambda x: _safe_float(x.get("value")))
            return f"Top region is {top.get('region')} contributing {top.get('pct', 0):.2f}%."
    if any(x in p for x in ["category"]):
        tbl = c.get("category_contribution", {})
        labels = tbl.get("labels", [])
        vals = tbl.get("cmtd", [])
        if labels and vals:
            i = int(np.argmax(vals))
            return f"Top category is {labels[i]} with CMTD {_format_value(vals[i])}."
    if any(x in p for x in ["outlet", "productive"]):
        return f"Visited outlets: {k.get('visited_outlets', 0):,}."
    return "I can summarize MTD/LE, region leaders, category contribution, and outlet productivity from this uploaded workbook."


def _parse_json_like(text):
    if not text:
        return None
    if isinstance(text, dict):
        return text
    try:
        return json.loads(str(text))
    except Exception:
        pass
    raw = str(text)
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s : e + 1])
        except Exception:
            return None
    return None


def _fallback_chart_plan(prompt_text):
    p = str(prompt_text or "").lower()
    chart_type = "bar"
    if any(t in p for t in ["trend", "monthly", "daily", "week", "line"]):
        chart_type = "line"
    elif any(t in p for t in ["pie", "share", "contribution %", "mix"]):
        chart_type = "pie"
    elif "heatmap" in p:
        chart_type = "heatmap"

    dim = "category"
    if "region" in p:
        dim = "region"
    elif "channel" in p:
        dim = "business_channel"
    elif "brand" in p:
        dim = "brand"
    elif "account" in p:
        dim = "account_group"
    elif any(t in p for t in ["month", "date", "daily", "weekly", "trend"]):
        dim = "date"

    metric = "value"
    if "volume" in p or "qty" in p or "quantity" in p:
        metric = "volume"
    if "outlet" in p and "count" in p:
        metric = "count_outlets"

    return {
        "title": "Custom Chart",
        "type": chart_type,
        "dimension": dim,
        "secondary_dimension": "region" if chart_type == "heatmap" and dim != "region" else "",
        "value_metric": metric,
        "aggregation": "sum",
        "time_grain": "month",
        "limit": 12,
        "sort": "desc",
    }


def _dim_to_col(dim_name):
    d = str(dim_name or "").strip().lower()
    mapping = {
        "account_group": "_account_group",
        "bcra_dairy": "_bcra_dairy",
        "brand": "_brand",
        "region": "_region",
        "business_channel": "_business_channel",
        "category": "_category",
        "outlet": "_outlet",
        "productive_flag": "_productive_flag",
        "date": "_date",
    }
    return mapping.get(d)


def _aggregate_series(df, dim_col, metric_col, agg="sum", limit=12, sort="desc", time_grain="month"):
    if dim_col == "_date":
        s = pd.to_datetime(df["_date"], errors="coerce")
        tg = str(time_grain or "month").strip().lower()
        freq = "M"
        if tg == "week":
            freq = "W"
        elif tg == "day":
            freq = "D"
        x = s.dt.to_period(freq).astype(str)
        grp = pd.DataFrame({"x": x, "y": pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)})
        out = grp.groupby("x", as_index=False)["y"].sum().sort_values("x")
        return out["x"].tolist(), out["y"].astype(float).tolist()

    temp = df[[dim_col, metric_col]].copy()
    temp[dim_col] = temp[dim_col].astype(str).fillna("Unknown")
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce").fillna(0.0)
    if agg == "avg":
        out = temp.groupby(dim_col, as_index=False)[metric_col].mean()
    elif agg == "count":
        out = temp.groupby(dim_col, as_index=False)[metric_col].count()
    else:
        out = temp.groupby(dim_col, as_index=False)[metric_col].sum()

    ascending = str(sort or "").lower() != "desc"
    out = out.sort_values(metric_col, ascending=ascending).head(max(1, int(limit or 12)))
    return out[dim_col].astype(str).tolist(), out[metric_col].astype(float).tolist()


def _heatmap_data(df, x_dim, y_dim, metric_col, limit=12):
    t = df[[x_dim, y_dim, metric_col]].copy()
    t[x_dim] = t[x_dim].astype(str).fillna("Unknown")
    t[y_dim] = t[y_dim].astype(str).fillna("Unknown")
    t[metric_col] = pd.to_numeric(t[metric_col], errors="coerce").fillna(0.0)
    pv = t.pivot_table(index=y_dim, columns=x_dim, values=metric_col, aggfunc="sum", fill_value=0.0)
    if pv.empty:
        return [], [], []
    pv = pv.iloc[: max(1, int(limit or 12)), : max(1, int(limit or 12))]
    return (
        [str(c) for c in pv.columns.tolist()],
        [str(i) for i in pv.index.tolist()],
        [[float(v) for v in row] for row in pv.to_numpy().tolist()],
    )


def _dim_sql_expr(dim_col, time_grain="month"):
    if dim_col == "_date":
        tg = str(time_grain or "month").strip().lower()
        if tg == "day":
            fmt = "%Y-%m-%d"
        elif tg == "week":
            fmt = "%Y-W%W"
        else:
            fmt = "%Y-%m"
        return f"strftime(CAST({_sql_ident('_date')} AS DATE), '{fmt}')"
    return f"CAST({_sql_ident(dim_col)} AS VARCHAR)"


def _metric_sql_expr(value_metric, agg, metric_col):
    if value_metric == "count_outlets":
        return f"COUNT(DISTINCT CAST({_sql_ident('_outlet')} AS VARCHAR))"
    base = f"COALESCE(CAST({_sql_ident(metric_col)} AS DOUBLE), 0.0)"
    if agg == "avg":
        return f"AVG({base})"
    if agg == "count":
        return "COUNT(1)"
    return f"SUM({base})"


def _build_chart_from_spec_duckdb(work_df, spec, chart_id=None, title_override=None):
    if duckdb is None:
        raise RuntimeError("duckdb not available")
    plan = spec if isinstance(spec, dict) else {}
    chart_type = str(plan.get("type") or "bar").strip().lower()
    if chart_type not in {"bar", "line", "scatter", "pie", "heatmap", "table"}:
        chart_type = "bar"
    dim = str(plan.get("dimension") or "category").strip().lower()
    secondary_dim = str(plan.get("secondary_dimension") or "").strip().lower()
    value_metric = str(plan.get("value_metric") or "value").strip().lower()
    agg = str(plan.get("aggregation") or "sum").strip().lower()
    if agg not in {"sum", "avg", "count"}:
        agg = "sum"
    limit = max(1, min(30, int(plan.get("limit") or 12)))
    sort = "asc" if str(plan.get("sort") or "").strip().lower() == "asc" else "desc"
    time_grain = str(plan.get("time_grain") or "month").strip().lower()
    if time_grain not in {"day", "week", "month"}:
        time_grain = "month"
    title = str(title_override or plan.get("title") or "Custom Chart").strip() or "Custom Chart"

    metric_col = "_value_metric"
    if value_metric == "volume":
        metric_col = "_volume_metric"
    elif value_metric == "count_outlets":
        metric_col = "_outlet"
        agg = "count"

    dim_col = _dim_to_col(dim) or "_category"
    chart = {
        "id": str(chart_id or f"custom_{int(datetime.utcnow().timestamp() * 1000)}"),
        "title": title,
        "type": chart_type,
        "x": [],
        "y": [],
        "z": [],
        "xlabel": dim.replace("_", " ").title(),
        "ylabel": value_metric.replace("_", " ").title(),
        "spec": {
            "dimension": dim,
            "secondary_dimension": secondary_dim,
            "value_metric": value_metric,
            "aggregation": agg,
            "limit": limit,
            "sort": sort,
            "type": chart_type,
            "title": title,
            "time_grain": time_grain,
        },
    }

    conn = duckdb.connect(database=":memory:")
    try:
        safe_df = _normalize_cockpit_df(work_df)
        conn.register("analysis_view", safe_df)
        x_expr = _dim_sql_expr(dim_col, time_grain=time_grain)
        metric_expr = _metric_sql_expr(value_metric, agg, metric_col)

        if chart_type == "heatmap":
            y_dim_col = _dim_to_col(secondary_dim) if secondary_dim else "_region"
            y_expr = _dim_sql_expr(y_dim_col)
            sql = (
                f"SELECT {x_expr} AS x, {y_expr} AS y, {metric_expr} AS z "
                "FROM analysis_view GROUP BY 1, 2 ORDER BY 1, 2"
            )
            raw = conn.execute(sql).fetchdf()
            if raw.empty:
                chart["type"] = "bar"
                chart["spec"]["type"] = "bar"
            else:
                pivot = raw.pivot_table(index="y", columns="x", values="z", aggfunc="sum", fill_value=0.0)
                pivot = pivot.iloc[: max(1, limit), : max(1, limit)]
                chart["x"] = [str(c) for c in pivot.columns.tolist()]
                chart["y"] = [str(i) for i in pivot.index.tolist()]
                chart["z"] = [[float(v) for v in row] for row in pivot.to_numpy().tolist()]
                chart["xlabel"] = str(dim).replace("_", " ").title()
                chart["ylabel"] = str((secondary_dim or "region")).replace("_", " ").title()
                return chart

        # Multi-series support for "date by region/channel/..." style requests.
        if chart_type in {"line", "bar", "scatter"} and secondary_dim:
            sec_dim_col = _dim_to_col(secondary_dim)
            if sec_dim_col:
                sec_expr = _dim_sql_expr(sec_dim_col, time_grain=time_grain)
                sql_series = (
                    f"SELECT {x_expr} AS x, {sec_expr} AS s, {metric_expr} AS y "
                    "FROM analysis_view GROUP BY 1, 2 ORDER BY 1, 2"
                )
                raw = conn.execute(sql_series).fetchdf()
                if not raw.empty:
                    raw["x"] = raw["x"].astype(str)
                    raw["s"] = raw["s"].astype(str)
                    raw["y"] = pd.to_numeric(raw["y"], errors="coerce").fillna(0.0)
                    top_series = (
                        raw.groupby("s", as_index=False)["y"]
                        .sum()
                        .sort_values("y", ascending=False)
                        .head(max(1, min(limit, 8)))["s"]
                        .tolist()
                    )
                    raw = raw[raw["s"].isin(top_series)]
                    x_values = sorted(raw["x"].dropna().unique().tolist())
                    if dim_col == "_date" and len(x_values) <= 1:
                        # If only one time bucket exists, render a category comparison instead.
                        fallback = (
                            raw.groupby("s", as_index=False)["y"]
                            .sum()
                            .sort_values("y", ascending=(sort == "asc"))
                            .head(max(1, limit))
                        )
                        chart["type"] = "bar"
                        chart["spec"]["type"] = "bar"
                        chart["x"] = fallback["s"].astype(str).tolist()
                        chart["y"] = [float(v) for v in fallback["y"].tolist()]
                        chart["xlabel"] = str(secondary_dim).replace("_", " ").title()
                        chart["ylabel"] = value_metric.replace("_", " ").title()
                        chart["series"] = []
                        return chart
                    series_out = []
                    for s_name in top_series:
                        part = raw[raw["s"] == s_name].sort_values("x")
                        series_out.append(
                            {
                                "name": str(s_name),
                                "x": part["x"].astype(str).tolist(),
                                "y": [float(v) for v in part["y"].tolist()],
                            }
                        )
                    chart["series"] = series_out
                    chart["x"] = x_values
                    chart["y"] = []
                    chart["xlabel"] = str(dim).replace("_", " ").title()
                    chart["ylabel"] = value_metric.replace("_", " ").title()
                    return chart

        order_sql = "ORDER BY 1 ASC" if dim_col == "_date" else f"ORDER BY 2 {'ASC' if sort == 'asc' else 'DESC'}"
        sql = (
            f"SELECT {x_expr} AS x, {metric_expr} AS y "
            f"FROM analysis_view GROUP BY 1 {order_sql} LIMIT {int(limit)}"
        )
        out = conn.execute(sql).fetchdf()
        chart["x"] = [str(v) for v in out["x"].tolist()] if "x" in out else []
        chart["y"] = [float(v) for v in out["y"].tolist()] if "y" in out else []
        if chart_type == "table":
            chart["columns"] = [chart["xlabel"], chart["ylabel"]]
            chart["rows"] = [[xv, float(yv)] for xv, yv in zip(chart["x"], chart["y"])]
        return chart
    finally:
        conn.close()


def _build_chart_from_spec(work_df, spec, chart_id=None, title_override=None):
    try:
        if duckdb is not None:
            return _build_chart_from_spec_duckdb(
                work_df=work_df,
                spec=spec,
                chart_id=chart_id,
                title_override=title_override,
            )
    except Exception:
        # Graceful fallback to pandas path.
        pass
    plan = spec if isinstance(spec, dict) else {}
    chart_type = str(plan.get("type") or "bar").strip().lower()
    if chart_type not in {"bar", "line", "scatter", "pie", "heatmap", "table"}:
        chart_type = "bar"
    dim = str(plan.get("dimension") or "category").strip().lower()
    secondary_dim = str(plan.get("secondary_dimension") or "").strip().lower()
    value_metric = str(plan.get("value_metric") or "value").strip().lower()
    agg = str(plan.get("aggregation") or "sum").strip().lower()
    if agg not in {"sum", "avg", "count"}:
        agg = "sum"
    limit = max(1, min(30, int(plan.get("limit") or 12)))
    sort = "asc" if str(plan.get("sort") or "").strip().lower() == "asc" else "desc"
    time_grain = str(plan.get("time_grain") or "month").strip().lower()
    if time_grain not in {"day", "week", "month"}:
        time_grain = "month"
    title = str(title_override or plan.get("title") or "Custom Chart").strip() or "Custom Chart"

    metric_col = "_value_metric"
    if value_metric == "volume":
        metric_col = "_volume_metric"
    elif value_metric == "count_outlets":
        metric_col = "_outlet"
        agg = "count"

    dim_col = _dim_to_col(dim) or "_category"
    chart = {
        "id": str(chart_id or f"custom_{int(datetime.utcnow().timestamp() * 1000)}"),
        "title": title,
        "type": chart_type,
        "x": [],
        "y": [],
        "z": [],
        "xlabel": dim.replace("_", " ").title(),
        "ylabel": value_metric.replace("_", " ").title(),
        "spec": {
            "dimension": dim,
            "secondary_dimension": secondary_dim,
            "value_metric": value_metric,
            "aggregation": agg,
            "limit": limit,
            "sort": sort,
            "type": chart_type,
            "title": title,
            "time_grain": time_grain,
        },
    }

    if chart_type == "heatmap":
        x_dim = dim_col
        y_dim = _dim_to_col(secondary_dim) if secondary_dim else "_region"
        temp = work_df
        metric_for_heat = metric_col
        if x_dim == "_date":
            temp = work_df.copy()
            freq = "M"
            if time_grain == "week":
                freq = "W"
            elif time_grain == "day":
                freq = "D"
            temp["_date_bucket"] = pd.to_datetime(temp["_date"], errors="coerce").dt.to_period(freq).astype(str)
            x_dim = "_date_bucket"
        if value_metric == "count_outlets":
            temp = temp.copy()
            temp["_count"] = 1.0
            metric_for_heat = "_count"
        x, y, z = _heatmap_data(temp, x_dim, y_dim, metric_for_heat, limit=limit)
        if x and y:
            chart["x"], chart["y"], chart["z"] = x, y, z
            chart["xlabel"] = str(dim).replace("_", " ").title()
            chart["ylabel"] = str((secondary_dim or "region")).replace("_", " ").title()
            return chart
        chart["type"] = "bar"
        chart_type = "bar"
        chart["spec"]["type"] = "bar"

    source_df = work_df.assign(_count=1.0) if value_metric == "count_outlets" else work_df

    if chart_type in {"line", "bar", "scatter"} and secondary_dim:
        sec_col = _dim_to_col(secondary_dim)
        if sec_col:
            temp = source_df[[dim_col, sec_col, "_count" if value_metric == "count_outlets" else metric_col]].copy()
            metric_use = "_count" if value_metric == "count_outlets" else metric_col
            if dim_col == "_date":
                freq = "M"
                if time_grain == "week":
                    freq = "W"
                elif time_grain == "day":
                    freq = "D"
                temp["_x"] = pd.to_datetime(temp[dim_col], errors="coerce").dt.to_period(freq).astype(str)
            else:
                temp["_x"] = temp[dim_col].astype(str)
            temp["_s"] = temp[sec_col].astype(str)
            temp["_m"] = pd.to_numeric(temp[metric_use], errors="coerce").fillna(0.0)
            grouped = temp.groupby(["_x", "_s"], as_index=False)["_m"].sum()
            top_series = (
                grouped.groupby("_s", as_index=False)["_m"]
                .sum()
                .sort_values("_m", ascending=False)
                .head(max(1, min(limit, 8)))["_s"]
                .tolist()
            )
            grouped = grouped[grouped["_s"].isin(top_series)]
            x_values = sorted(grouped["_x"].dropna().unique().tolist())
            if dim_col == "_date" and len(x_values) <= 1:
                fallback = (
                    grouped.groupby("_s", as_index=False)["_m"]
                    .sum()
                    .sort_values("_m", ascending=(sort == "asc"))
                    .head(max(1, limit))
                )
                chart["type"] = "bar"
                chart["spec"]["type"] = "bar"
                chart["x"] = fallback["_s"].astype(str).tolist()
                chart["y"] = [float(v) for v in fallback["_m"].tolist()]
                chart["xlabel"] = str(secondary_dim).replace("_", " ").title()
                chart["ylabel"] = value_metric.replace("_", " ").title()
                chart["series"] = []
                return chart
            series_out = []
            for s_name in top_series:
                part = grouped[grouped["_s"] == s_name].sort_values("_x")
                series_out.append(
                    {
                        "name": str(s_name),
                        "x": part["_x"].astype(str).tolist(),
                        "y": [float(v) for v in part["_m"].tolist()],
                    }
                )
            chart["series"] = series_out
            chart["x"] = x_values
            chart["y"] = []
            chart["xlabel"] = str(dim).replace("_", " ").title()
            chart["ylabel"] = value_metric.replace("_", " ").title()
            return chart

    x, y = _aggregate_series(
        source_df,
        dim_col,
        "_count" if value_metric == "count_outlets" else metric_col,
        agg=agg,
        limit=limit,
        sort=sort,
        time_grain=time_grain,
    )
    chart["x"], chart["y"] = x, y
    if chart_type == "table":
        chart["columns"] = [chart["xlabel"], chart["ylabel"]]
        chart["rows"] = [[xv, float(yv)] for xv, yv in zip(x, y)]
    return chart


def _resolve_cockpit_month_window(connection, invoice_fqn, selected_month=""):
    max_date_raw = _db_scalar(
        connection,
        f"SELECT MAX(CAST({_quote_db_ident('Date')} AS DATE)) AS max_date FROM {invoice_fqn}",
        default=None,
    )
    max_date_ts = pd.to_datetime(max_date_raw, errors="coerce")
    if pd.isna(max_date_ts):
        raise ValueError("Could not resolve max Date from invoice table.")
    max_date_ts = pd.Timestamp(max_date_ts).normalize()

    selected_raw = str(selected_month or "").strip()
    selected_match = re.match(r"^\d{4}-\d{2}$", selected_raw)
    if selected_match:
        selected_anchor = pd.to_datetime(selected_raw + "-01", errors="coerce")
        if pd.isna(selected_anchor):
            selected_anchor = max_date_ts.replace(day=1)
        else:
            selected_anchor = pd.Timestamp(selected_anchor).normalize()
    else:
        selected_anchor = max_date_ts.replace(day=1)

    curr_start = selected_anchor
    selected_key = selected_anchor.strftime("%Y-%m")
    max_key = max_date_ts.strftime("%Y-%m")
    month_last_day = monthrange(selected_anchor.year, selected_anchor.month)[1]
    full_month_end = selected_anchor.replace(day=month_last_day)
    curr_end = max_date_ts if selected_key == max_key else full_month_end
    return curr_start, curr_end, selected_key


def _validate_cockpit_custom_sql(sql_text, invoice_table, invoice_fqn):
    sql = str(sql_text or "").strip()
    if not sql:
        raise ValueError("LLM returned empty SQL.")
    if not sql.lower().startswith("select"):
        raise ValueError("Generated SQL must start with SELECT.")
    if ";" in sql.rstrip(";"):
        raise ValueError("Generated SQL must be a single statement.")
    low = sql.lower()
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " truncate ", " merge ", " create ", " grant ", " revoke "]
    if any(tok in f" {low} " for tok in forbidden):
        raise ValueError("Generated SQL contains forbidden keywords.")

    invoice_table_low = str(invoice_table or "").strip().lower()
    invoice_fqn_low = str(invoice_fqn or "").strip().lower()
    sql_compact = low.replace("`", "")
    if invoice_table_low and invoice_table_low not in sql_compact and invoice_fqn_low.replace("`", "") not in sql_compact:
        raise ValueError("Generated SQL must reference the configured invoice table.")
    return sql


def _pick_df_col(result_df, preferred_name, fallback_idx):
    if result_df is None or result_df.empty:
        return None
    cols = [str(c) for c in result_df.columns]
    norm_map = {_norm(c): c for c in cols}
    pref = _norm(preferred_name)
    if pref and pref in norm_map:
        return norm_map[pref]
    if 0 <= fallback_idx < len(cols):
        return cols[fallback_idx]
    return cols[0] if cols else None


def _build_cockpit_chart_from_query_result(result_df, chart_plan, sql_text, chart_id=None):
    plan = chart_plan if isinstance(chart_plan, dict) else {}
    chart_type = str(plan.get("type") or "bar").strip().lower()
    if chart_type not in {"bar", "line", "scatter", "pie", "heatmap", "table"}:
        chart_type = "bar"
    title = str(plan.get("title") or "Custom Chart").strip() or "Custom Chart"
    xlabel = str(plan.get("xlabel") or "X").strip() or "X"
    ylabel = str(plan.get("ylabel") or "Y").strip() or "Y"
    x_col_hint = str(plan.get("x_col") or "x").strip()
    y_col_hint = str(plan.get("y_col") or "y").strip()
    z_col_hint = str(plan.get("z_col") or "z").strip()

    chart = {
        "id": str(chart_id or f"custom_{int(datetime.utcnow().timestamp() * 1000)}"),
        "title": title,
        "type": chart_type,
        "x": [],
        "y": [],
        "z": [],
        "xlabel": xlabel,
        "ylabel": ylabel,
        "sql": str(sql_text or ""),
        "spec": {
            "type": chart_type,
            "title": title,
            "x_col": x_col_hint,
            "y_col": y_col_hint,
            "z_col": z_col_hint,
            "source_mode": "databricks",
        },
    }

    if result_df is None or result_df.empty:
        return chart

    df_out = result_df.copy()
    x_col = _pick_df_col(df_out, x_col_hint, 0)
    y_col = _pick_df_col(df_out, y_col_hint, 1)
    z_col = _pick_df_col(df_out, z_col_hint, 2)

    if chart_type == "heatmap":
        if x_col and y_col and z_col and z_col in df_out.columns:
            pv = df_out.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="sum", fill_value=0.0)
            chart["x"] = [str(c) for c in pv.columns.tolist()]
            chart["y"] = [str(i) for i in pv.index.tolist()]
            chart["z"] = [[float(v) for v in row] for row in pv.to_numpy().tolist()]
            return chart
        chart["type"] = "bar"

    if chart["type"] == "table":
        chart["columns"] = [str(c) for c in df_out.columns]
        chart["rows"] = [[r.get(c) for c in df_out.columns] for _, r in df_out.iterrows()]
        return chart

    if not x_col:
        return chart
    if not y_col:
        y_col = x_col

    chart["x"] = df_out[x_col].astype(str).tolist()
    chart["y"] = [float(_safe_float(v)) for v in pd.to_numeric(df_out[y_col], errors="coerce").fillna(0.0).tolist()]
    return chart


def _generate_cockpit_custom_chart_databricks(
    user_prompt,
    active_filters=None,
    existing_chart=None,
    connection=None,
    invoice_table=DEFAULT_COCKPIT_INVOICE_TABLE,
    target_table=DEFAULT_COCKPIT_TARGET_TABLE,
    selected_month="",
):
    prompt = str(user_prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    invoice_table = str(invoice_table or DEFAULT_COCKPIT_INVOICE_TABLE).strip()
    target_table = str(target_table or DEFAULT_COCKPIT_TARGET_TABLE).strip()
    invoice_fqn = _quote_db_table_fqn(invoice_table)
    target_fqn = _quote_db_table_fqn(target_table)
    filters = active_filters if isinstance(active_filters, dict) else {}
    logs = []

    own_connection = False
    if connection is None:
        connection = get_databricks_connection()
        own_connection = True

    try:
        filter_conditions = _build_cockpit_filter_conditions(filters, table_kind="invoice")
        full_where = " AND ".join(list(filter_conditions or [])) if filter_conditions else "1=1"

        use_existing_spec_only = str(prompt).strip().lower() in {"__refresh__", "__use_existing_spec__", "__reuse_spec__"}
        llm_plan = None

        if use_existing_spec_only and isinstance(existing_chart, dict) and str(existing_chart.get("sql") or "").strip():
            llm_plan = {
                "title": str(existing_chart.get("title") or "Custom Chart"),
                "type": str(existing_chart.get("type") or "bar"),
                "xlabel": str(existing_chart.get("xlabel") or "X"),
                "ylabel": str(existing_chart.get("ylabel") or "Y"),
                "x_col": str((existing_chart.get("spec") or {}).get("x_col") or "x"),
                "y_col": str((existing_chart.get("spec") or {}).get("y_col") or "y"),
                "z_col": str((existing_chart.get("spec") or {}).get("z_col") or "z"),
                "sql": str(existing_chart.get("sql") or "").strip(),
            }
        else:
            metadata_cols = [
                ("Date", "date"),
                ("Channel", "string"),
                ("Brand", "string"),
                ("Retailer", "string"),
                ("Region", "string"),
                ("State", "string"),
                ("Distributor", "string"),
                ("Category", "string"),
                ("Value", "double"),
                ("Volume", "double"),
            ]
            metadata_text = ", ".join([f"{c} ({t})" for c, t in metadata_cols])
            edit_note = ""
            if isinstance(existing_chart, dict) and existing_chart:
                edit_note = (
                    f"Existing chart to edit: title={existing_chart.get('title')}, "
                    f"type={existing_chart.get('type')}, "
                    f"xlabel={existing_chart.get('xlabel')}, ylabel={existing_chart.get('ylabel')}. "
                )

            prompt_text = f"""
You are a BI SQL chart planner for Databricks.
Primary table: {invoice_table}
Primary columns (metadata only): {metadata_text}
Secondary table: {target_table}
Secondary columns (metadata only): Monthly (yyyy-MM), Region, Channel, Brand, Category, State, Values_Target, Vol_Target

Join logic (use ONLY when needed):
- Join invoice -> target on:
  date_format(CAST(invoice.Date AS DATE), 'yyyy-MM') = SUBSTR(CAST(target.Monthly AS STRING), 1, 7)
  AND invoice.Region = target.Region
  AND invoice.Channel = target.Channel
  AND invoice.Brand = target.Brand
  AND invoice.Category = target.Category
- State MUST NOT be part of join keys.
- If prompt mentions target/MSP/achievement/vs target/gap to target/plan, use LEFT JOIN to target.
- Otherwise query invoice table only.

Use full available history by default.
Do NOT restrict SQL to a single month unless the user explicitly asks for a month/date range.
Active filter clause for invoice table:
{full_where}

Return ONLY JSON with:
{{
  "title": "Chart title",
  "type": "bar|line|scatter|pie|heatmap|table",
  "xlabel": "X axis label",
  "ylabel": "Y axis label",
  "x_col": "x",
  "y_col": "y",
  "z_col": "z for heatmap else empty",
  "sql": "SELECT ... FROM {invoice_table} WHERE ... "
}}

Rules:
- SQL must start with SELECT.
- SQL must reference table {invoice_table}.
- If active filters are present, include them in invoice WHERE.
- Return query columns as aliases x and y (and z for heatmap) whenever possible.
- No DML/DDL (no INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/MERGE/CREATE).
{edit_note}
User request: {prompt}
""".strip()

            llm_raw, _ = call_ai_with_retry(
                messages=[{"role": "user", "content": prompt_text}],
                json_mode=True,
                retries=2,
                debug_logs=logs,
                context="Generate Cockpit Databricks Custom Chart SQL",
            )
            llm_plan = _parse_json_like(llm_raw)

        if not isinstance(llm_plan, dict):
            logs.append("[FALLBACK] Using deterministic Databricks cockpit custom chart SQL.")
            llm_plan = {
                "title": "Monthly Value Trend",
                "type": "line",
                "xlabel": "Date",
                "ylabel": "Value",
                "x_col": "x",
                "y_col": "y",
                "z_col": "",
                "sql": (
                    f"SELECT CAST({_quote_db_ident('Date')} AS DATE) AS x, "
                    f"SUM(COALESCE(CAST({_quote_db_ident('Value')} AS DOUBLE), 0.0)) AS y "
                    f"FROM {invoice_fqn} WHERE {full_where} "
                    f"GROUP BY 1 ORDER BY 1"
                ),
            }

        sql_text = _validate_cockpit_custom_sql(llm_plan.get("sql"), invoice_table=invoice_table, invoice_fqn=invoice_fqn)

        print(f"[COCKPIT CUSTOM CHART SQL] {sql_text}", flush=True)
        result_df = _run_db_query(connection, sql_text)

        chart = _build_cockpit_chart_from_query_result(
            result_df=result_df,
            chart_plan=llm_plan,
            sql_text=sql_text,
            chart_id=(existing_chart or {}).get("id") if isinstance(existing_chart, dict) else None,
        )
        chart["spec"] = {
            **(chart.get("spec") or {}),
            "source_mode": "databricks",
            "selected_month": str(selected_month or "__all_data__"),
            "invoice_table": invoice_table,
            "target_table": target_table,
            "x_col": str(llm_plan.get("x_col") or "x"),
            "y_col": str(llm_plan.get("y_col") or "y"),
            "z_col": str(llm_plan.get("z_col") or "z"),
        }
        return {"chart": chart, "logs": logs}
    finally:
        if own_connection:
            try:
                connection.close()
            except Exception:
                pass


def generate_cockpit_custom_chart_from_prompt(
    df,
    user_prompt,
    active_filters=None,
    existing_chart=None,
    connection=None,
    invoice_table=DEFAULT_COCKPIT_INVOICE_TABLE,
    target_table=DEFAULT_COCKPIT_TARGET_TABLE,
    selected_month="",
    data_source="excel",
):
    if str(data_source or "").strip().lower() == "databricks":
        return _generate_cockpit_custom_chart_databricks(
            user_prompt=user_prompt,
            active_filters=active_filters,
            existing_chart=existing_chart,
            connection=connection,
            invoice_table=invoice_table,
            target_table=target_table,
            selected_month=selected_month,
        )

    if df is None or df.empty:
        raise ValueError("No data available for chart generation.")
    df = _normalize_cockpit_df(df)
    prompt = str(user_prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    filters = active_filters if isinstance(active_filters, dict) else {}
    work_df = _apply_filters(df, filters)
    if work_df.empty:
        work_df = df.copy()
    work_df = _normalize_cockpit_df(work_df)

    dataset_columns = [str(c) for c in df.columns if not str(c).startswith("_")]
    chart_types = ["bar", "line", "scatter", "pie", "heatmap", "table"]
    dims = [
        "account_group",
        "bcra_dairy",
        "brand",
        "region",
        "business_channel",
        "category",
        "outlet",
        "productive_flag",
        "date",
    ]
    metrics = ["value", "volume", "count_outlets"]

    logs = []
    use_existing_spec_only = (
        str(prompt).strip().lower() in {"__refresh__", "__use_existing_spec__", "__reuse_spec__"}
    )
    if use_existing_spec_only and isinstance(existing_chart, dict) and isinstance(existing_chart.get("spec"), dict):
        chart = _build_chart_from_spec(
            work_df,
            existing_chart.get("spec"),
            chart_id=existing_chart.get("id"),
            title_override=existing_chart.get("title"),
        )
        return {"chart": chart, "logs": logs}
    edit_note = ""
    if isinstance(existing_chart, dict) and existing_chart:
        edit_note = (
            f"Existing chart to edit: title={existing_chart.get('title')}, "
            f"type={existing_chart.get('type')}, "
            f"dimension={existing_chart.get('spec', {}).get('dimension')}, "
            f"metric={existing_chart.get('spec', {}).get('value_metric')}"
        )

    prompt_text = f"""
You are a BI chart planner for a Sales Cockpit.
Build one chart plan from a user request.

DATASET METADATA (column names only):
{", ".join(dataset_columns)}

Return ONLY JSON with this shape:
{{
  "title": "Chart title",
  "type": "bar|line|scatter|pie|heatmap|table",
  "dimension": "account_group|bcra_dairy|brand|region|business_channel|category|outlet|productive_flag|date",
  "secondary_dimension": "optional for heatmap from same dimension list, else empty",
  "value_metric": "value|volume|count_outlets",
  "aggregation": "sum|avg|count",
  "time_grain": "day|week|month",
  "limit": 12,
  "sort": "desc|asc"
}}

Rules:
- Pick exactly one chart.
- Use only allowed enums.
- Prefer readable business chart titles.
- If user asks trend, use dimension=date.
- For share/composition, prefer pie.

{edit_note}
USER REQUEST:
{prompt}
""".strip()

    plan = None
    try:
        llm_raw, _ = call_ai_with_retry(
            messages=[{"role": "user", "content": prompt_text}],
            json_mode=True,
            retries=2,
            debug_logs=logs,
            context="Generate Cockpit Custom Chart Plan",
        )
        plan = _parse_json_like(llm_raw)
    except Exception as e:
        logs.append(f"[LLM ERROR] Generate Cockpit Custom Chart Plan: {str(e)}")
    if not isinstance(plan, dict):
        logs.append("[FALLBACK] Using deterministic cockpit chart planner")
        plan = _fallback_chart_plan(prompt)

    chart_type = str(plan.get("type") or "bar").strip().lower()
    if chart_type not in chart_types:
        chart_type = "bar"
    dim = str(plan.get("dimension") or "category").strip().lower()
    if dim not in dims:
        dim = "category"
    secondary_dim = str(plan.get("secondary_dimension") or "").strip().lower()
    if secondary_dim and secondary_dim not in dims:
        secondary_dim = ""
    value_metric = str(plan.get("value_metric") or "value").strip().lower()
    if value_metric not in metrics:
        value_metric = "value"
    agg = str(plan.get("aggregation") or "sum").strip().lower()
    if agg not in {"sum", "avg", "count"}:
        agg = "sum"
    limit = max(1, min(30, int(plan.get("limit") or 12)))
    sort = "asc" if str(plan.get("sort") or "").strip().lower() == "asc" else "desc"
    title = str(plan.get("title") or "Custom Chart").strip() or "Custom Chart"

    metric_col = "_value_metric"
    if value_metric == "volume":
        metric_col = "_volume_metric"
    elif value_metric == "count_outlets":
        metric_col = "_outlet"
        agg = "count"

    dim_col = _dim_to_col(dim)
    if not dim_col:
        dim_col = "_category"
        dim = "category"

    chart = _build_chart_from_spec(
        work_df,
        {
            "dimension": dim,
            "secondary_dimension": secondary_dim,
            "value_metric": value_metric,
            "aggregation": agg,
            "limit": limit,
            "sort": sort,
            "type": chart_type,
            "title": title,
        },
    )
    return {"chart": chart, "logs": logs}

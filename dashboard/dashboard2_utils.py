import os
import math
from functools import lru_cache
from django.conf import settings

DASHBOARD2_SESSION_EXCEL_KEY = "dashboard2_excel_path"
DASHBOARD2_DEFAULT_XLSX = "Sales_Dashboard_Data_Dashboard_2_.xlsx"


def _dashboard2_default_excel_path():
    return os.path.join(settings.BASE_DIR, "dashboard", "static", "dashboard", DASHBOARD2_DEFAULT_XLSX)


def _dashboard2_resolve_excel_path(request):
    session_path = str(request.session.get(DASHBOARD2_SESSION_EXCEL_KEY) or "").strip()
    if session_path and os.path.exists(session_path):
        return session_path
    return _dashboard2_default_excel_path()


@lru_cache(maxsize=8)
def _dashboard2_read_excel_cached(excel_path, mtime):
    import pandas as pd
    xl = pd.ExcelFile(excel_path)
    return {
        "raw": xl.parse("Raw Data"),
        "brand_monthly": xl.parse("Brand Monthly Summary"),
        "channel_branch": xl.parse("Channel Branch Summary"),
        "state_district": xl.parse("State District Summary"),
        "brand_funnel": xl.parse("Brand Funnel"),
        "growth_drivers": xl.parse("Growth Drivers"),
    }


def _dashboard2_load_sheets(excel_path):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    mtime = os.path.getmtime(excel_path)
    cached = _dashboard2_read_excel_cached(excel_path, mtime)
    return {k: v.copy() for k, v in cached.items()}


def _dashboard2_to_num(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, float) and not math.isfinite(value):
            return float(default)
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _dashboard2_clean_records(df):
    import pandas as pd
    clean = []
    if df is None or df.empty:
        return clean
    for row in df.to_dict("records"):
        item = {}
        for k, v in row.items():
            if isinstance(v, float):
                item[k] = v if math.isfinite(v) else None
            elif hasattr(v, "item") and callable(getattr(v, "item", None)):
                try:
                    raw = v.item()
                    if isinstance(raw, float) and not math.isfinite(raw):
                        item[k] = None
                    else:
                        item[k] = raw
                except Exception:
                    item[k] = None
            elif pd.isna(v):
                item[k] = None
            else:
                item[k] = v
        clean.append(item)
    return clean


def _dashboard2_prepare_payload(excel_path, query_params):
    import pandas as pd

    sheets = _dashboard2_load_sheets(excel_path)
    raw = sheets["raw"]
    brand_monthly = sheets["brand_monthly"]
    channel_branch = sheets["channel_branch"]
    state_district = sheets["state_district"]
    brand_funnel = sheets["brand_funnel"]
    growth_drivers = sheets["growth_drivers"]

    # Normalize month/date columns for consistent filtering/sorting.
    if "Month_Date" in raw.columns:
        raw["Month_Date"] = pd.to_datetime(raw["Month_Date"], errors="coerce")
    if "Month" in raw.columns and "Month_Date" in raw.columns:
        raw["Month"] = raw["Month"].astype(str).str.strip()

    month_order = []
    if "Month_Date" in raw.columns and "Month" in raw.columns:
        month_ref = (
            raw.dropna(subset=["Month_Date"])
            .sort_values("Month_Date")[["Month", "Month_Date"]]
            .drop_duplicates("Month")
        )
        month_order = month_ref["Month"].astype(str).tolist()
    if not month_order:
        month_order = sorted(raw.get("Month", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    month_index = {m: idx for idx, m in enumerate(month_order)}

    def month_in_range(series, month_from, month_to):
        if not month_from and not month_to:
            return pd.Series([True] * len(series), index=series.index)
        start = month_index.get(month_from, 0)
        end = month_index.get(month_to, len(month_order) - 1)
        if start > end:
            start, end = end, start
        return series.astype(str).map(lambda m: month_index.get(m, -1)).between(start, end)

    # Filters
    sales_type = str(query_params.get("sales_type") or "Primary").strip()
    brand_filter = str(query_params.get("brand") or "All").strip()
    channel_filter = str(query_params.get("channel") or "All").strip()
    branch_filter = str(query_params.get("branch") or "All").strip()
    category_filter = str(query_params.get("category") or "All").strip()
    sub_brand_filter = str(query_params.get("sub_brand") or "All").strip()
    period_filter = str(query_params.get("period") or "MTD").strip()
    month_from = str(query_params.get("month_from") or "").strip()
    month_to = str(query_params.get("month_to") or "").strip()

    # Shared scope filters (everything except Sales_Type) so we can compute
    # Primary and Secondary KPI/trend blocks in parallel from the same context.
    df_scope = raw.copy()
    if brand_filter and brand_filter != "All" and "Brand" in df_scope.columns:
        df_scope = df_scope[df_scope["Brand"].astype(str) == brand_filter]
    if channel_filter and channel_filter != "All" and "Channel" in df_scope.columns:
        df_scope = df_scope[df_scope["Channel"].astype(str) == channel_filter]
    if branch_filter and branch_filter != "All" and "Branch_Name" in df_scope.columns:
        df_scope = df_scope[df_scope["Branch_Name"].astype(str) == branch_filter]
    if category_filter and category_filter != "All" and "Category" in df_scope.columns:
        df_scope = df_scope[df_scope["Category"].astype(str) == category_filter]
    if sub_brand_filter and sub_brand_filter != "All" and "Sub_Brand" in df_scope.columns:
        df_scope = df_scope[df_scope["Sub_Brand"].astype(str) == sub_brand_filter]
    if period_filter and period_filter != "All" and "MTD_YTD" in df_scope.columns:
        df_scope = df_scope[df_scope["MTD_YTD"].astype(str) == period_filter]
    if "Month" in df_scope.columns:
        df_scope = df_scope[month_in_range(df_scope["Month"], month_from, month_to)]

    # Active chart scope respects current Sales_Type selection.
    df = df_scope.copy()
    if sales_type and sales_type != "All" and "Sales_Type" in df.columns:
        df = df[df["Sales_Type"].astype(str) == sales_type]

    total_actual = _dashboard2_to_num(df.get("Actual_Sales", pd.Series(dtype=float)).sum())
    total_ly = _dashboard2_to_num(df.get("LY_Sales", pd.Series(dtype=float)).sum())
    total_target = _dashboard2_to_num(df.get("Target", pd.Series(dtype=float)).sum())
    growth_pct = ((total_actual - total_ly) / total_ly) if total_ly else 0.0
    achievement_pct = (total_actual / total_target) if total_target else 0.0

    def _month_series(frame):
        if frame is None or frame.empty:
            return {
                "months": [],
                "actual": [],
                "ly": [],
                "target": [],
                "growth_pct": [],
                "achievement_pct": [],
            }
        out = frame.groupby("Month", as_index=False).agg(
            Actual_Sales=("Actual_Sales", "sum"),
            LY_Sales=("LY_Sales", "sum"),
            Target=("Target", "sum"),
            Growth_Pct=("Growth_Pct", "mean"),
            Achievement_Pct=("Achievement_Pct", "mean"),
        )
        out["Month_Order"] = out["Month"].astype(str).map(lambda m: month_index.get(m, 999))
        out = out.sort_values("Month_Order")
        return {
            "months": out["Month"].astype(str).tolist(),
            "actual": [_dashboard2_to_num(v) for v in out["Actual_Sales"].tolist()],
            "ly": [_dashboard2_to_num(v) for v in out["LY_Sales"].tolist()],
            "target": [_dashboard2_to_num(v) for v in out["Target"].tolist()],
            "growth_pct": [_dashboard2_to_num(v) for v in out["Growth_Pct"].tolist()],
            "achievement_pct": [_dashboard2_to_num(v) for v in out["Achievement_Pct"].tolist()],
        }

    def _kpi_block(frame):
        if frame is None or frame.empty:
            return {
                "actual_sales": 0.0,
                "ly_sales": 0.0,
                "target": 0.0,
                "growth_pct": 0.0,
                "achievement_pct": 0.0,
                "sparkline_months": [],
                "sparkline_actual": [],
            }
        actual = _dashboard2_to_num(frame.get("Actual_Sales", pd.Series(dtype=float)).sum())
        ly = _dashboard2_to_num(frame.get("LY_Sales", pd.Series(dtype=float)).sum())
        target = _dashboard2_to_num(frame.get("Target", pd.Series(dtype=float)).sum())
        growth = ((actual - ly) / ly) if ly else 0.0
        achievement = (actual / target) if target else 0.0
        spark = frame.groupby("Month", as_index=False).agg(Actual_Sales=("Actual_Sales", "sum"))
        spark["Month_Order"] = spark["Month"].astype(str).map(lambda m: month_index.get(m, 999))
        spark = spark.sort_values("Month_Order")
        return {
            "actual_sales": actual,
            "ly_sales": ly,
            "target": target,
            "growth_pct": growth,
            "achievement_pct": achievement,
            "sparkline_months": spark["Month"].astype(str).tolist(),
            "sparkline_actual": [_dashboard2_to_num(v) for v in spark["Actual_Sales"].tolist()],
        }

    # Global trend (kept for compatibility) + explicit per sales-type trends.
    bm = brand_monthly.copy()
    if brand_filter != "All" and "Brand" in bm.columns:
        bm = bm[bm["Brand"].astype(str) == brand_filter]
    if "Month" in bm.columns:
        bm = bm[bm["Month"].astype(str).map(lambda m: month_index.get(m, -1)) >= 0]
        if month_from or month_to:
            bm = bm[month_in_range(bm["Month"], month_from, month_to)]
    trend = _month_series(bm)

    primary_scope = (
        df_scope[df_scope["Sales_Type"].astype(str) == "Primary"]
        if "Sales_Type" in df_scope.columns else df_scope.copy()
    )
    secondary_scope = (
        df_scope[df_scope["Sales_Type"].astype(str) == "Secondary"]
        if "Sales_Type" in df_scope.columns else df_scope.copy()
    )
    kpis_by_sales_type = {
        "Primary": _kpi_block(primary_scope),
        "Secondary": _kpi_block(secondary_scope),
    }
    trend_by_sales_type = {
        "Primary": _month_series(primary_scope),
        "Secondary": _month_series(secondary_scope),
    }

    # Channel -> Branch
    cb = channel_branch.copy()
    if brand_filter != "All" and "Brand" in cb.columns:
        cb = cb[cb["Brand"].astype(str) == brand_filter]
    if channel_filter != "All" and "Channel" in cb.columns:
        cb = cb[cb["Channel"].astype(str) == channel_filter]
    channel_summary = cb.groupby("Channel", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        Growth_Pct=("Growth_Pct", "mean"),
        Achievement_Pct=("Achievement_Pct", "mean"),
    ).sort_values("Actual_Sales", ascending=False)
    branch_by_channel = {}
    for ch in cb.get("Channel", pd.Series(dtype=str)).dropna().astype(str).unique().tolist():
        cdf = cb[cb["Channel"].astype(str) == ch]
        branch_by_channel[ch] = _dashboard2_clean_records(
            cdf.groupby("Branch_Name", as_index=False).agg(
                Actual_Sales=("Actual_Sales", "sum"),
                Growth_Pct=("Growth_Pct", "mean"),
                Achievement_Pct=("Achievement_Pct", "mean"),
            )
        )

    branch_summary = df.groupby("Branch_Name", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
        Target=("Target", "sum"),
    ).sort_values("Actual_Sales", ascending=False)

    state_summary = df.groupby("State", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
        Growth_Pct=("Growth_Pct", "mean"),
    ).sort_values("Actual_Sales", ascending=False).head(15)

    district_summary = df.groupby("District", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
        Growth_Pct=("Growth_Pct", "mean"),
    ).sort_values("Actual_Sales", ascending=False).head(15)

    asm_summary = df.groupby("ASM_HQ_Town", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
    ).sort_values("Actual_Sales", ascending=False).head(12)

    so_summary = df.groupby("SO_HQ_Town", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
    ).sort_values("Actual_Sales", ascending=False).head(12)

    pops_summary = df.groupby("Pops_Strata", as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
    ).sort_values("Actual_Sales", ascending=False)

    brand_sub = df.groupby(["Brand", "Sub_Brand"], as_index=False).agg(
        Actual_Sales=("Actual_Sales", "sum"),
        LY_Sales=("LY_Sales", "sum"),
        Growth_Pct=("Growth_Pct", "mean"),
        Achievement_Pct=("Achievement_Pct", "mean"),
    ).sort_values("Actual_Sales", ascending=False).head(20)

    sd = state_district.copy()
    if branch_filter != "All" and "Branch_Name" in sd.columns:
        sd = sd[sd["Branch_Name"].astype(str) == branch_filter]
    sd_grouped = sd.groupby("State", as_index=False).agg(
        Sales=("Actual_Sales", "sum"),
        Salience_Pct=("Salience_Pct", "mean"),
        Growth_Contribution_Pct=("Growth_Contribution_Pct", "mean"),
        Growth_Pct=("Growth_Pct", "mean"),
    ).sort_values("Sales", ascending=False)

    bf = brand_funnel.copy()
    if brand_filter != "All" and "Brand" in bf.columns:
        bf = bf[bf["Brand"].astype(str) == brand_filter]
    if "Month" in bf.columns and (month_from or month_to):
        bf = bf[month_in_range(bf["Month"], month_from, month_to)]
    funnel_by_month = bf.groupby("Month", as_index=False).agg(
        GRPs=("GRPs", "mean"),
        SOV_Pct=("SOV_Pct", "mean"),
        TOM_Pct=("TOM_Pct", "mean"),
    )
    funnel_by_month["Month_Order"] = funnel_by_month["Month"].astype(str).map(lambda m: month_index.get(m, 999))
    funnel_by_month = funnel_by_month.sort_values("Month_Order")

    gd = growth_drivers.copy()
    if sales_type != "All" and "Sales_Type" in gd.columns:
        gd = gd[gd["Sales_Type"].astype(str) == sales_type]
    growth = gd[gd["Driver_Type"].astype(str) == "Growth Driver"].sort_values("Delta", ascending=False).head(5)
    decline = gd[gd["Driver_Type"].astype(str) == "Decline Contributor"].sort_values("Delta", ascending=True).head(5)

    filters = {
        "months": month_order,
        "brands": ["All"] + sorted(raw.get("Brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "sub_brands": ["All"] + sorted(raw.get("Sub_Brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "channels": ["All"] + sorted(raw.get("Channel", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "branches": ["All"] + sorted(raw.get("Branch_Name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "categories": ["All"] + sorted(raw.get("Category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "sales_types": ["Primary", "Secondary", "All"],
        "periods": ["MTD", "YTD", "All"],
    }

    payload = {
        "kpis": {
            "actual_sales": total_actual,
            "ly_sales": total_ly,
            "target": total_target,
            "growth_pct": growth_pct,
            "achievement_pct": achievement_pct,
        },
        "kpis_by_sales_type": kpis_by_sales_type,
        "trend": trend,
        "trend_by_sales_type": trend_by_sales_type,
        "channel_summary": _dashboard2_clean_records(channel_summary),
        "branch_by_channel": branch_by_channel,
        "branch_summary": _dashboard2_clean_records(branch_summary),
        "state_summary": _dashboard2_clean_records(state_summary),
        "district_summary": _dashboard2_clean_records(district_summary),
        "asm_summary": _dashboard2_clean_records(asm_summary),
        "so_summary": _dashboard2_clean_records(so_summary),
        "pops_summary": _dashboard2_clean_records(pops_summary),
        "brand_sub": _dashboard2_clean_records(brand_sub),
        "state_district": _dashboard2_clean_records(sd_grouped),
        "funnel": {
            "months": funnel_by_month["Month"].astype(str).tolist(),
            "grps": [_dashboard2_to_num(v) for v in funnel_by_month["GRPs"].tolist()],
            "sov": [_dashboard2_to_num(v) for v in funnel_by_month["SOV_Pct"].tolist()],
            "tom": [_dashboard2_to_num(v) for v in funnel_by_month["TOM_Pct"].tolist()],
        },
        "growth_drivers": _dashboard2_clean_records(growth),
        "decline_contributors": _dashboard2_clean_records(decline),
        "filters": filters,
        "applied_filters": {
            "sales_type": sales_type,
            "brand": brand_filter,
            "sub_brand": sub_brand_filter,
            "channel": channel_filter,
            "branch": branch_filter,
            "category": category_filter,
            "period": period_filter,
            "month_from": month_from or (month_order[0] if month_order else ""),
            "month_to": month_to or (month_order[-1] if month_order else ""),
        },
        "meta": {
            "excel_path": excel_path,
            "file_exists": os.path.exists(excel_path),
        },
    }
    return payload



# --- EXTRACtED VIEWS ---
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from .utils import _safe_json_response

@csrf_exempt
def dashboard2_upload(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    upload = request.FILES.get("file")
    if not upload:
        return _safe_json_response({"error": "Excel file is required"}, status=400)

    filename = str(getattr(upload, "name", "") or "").lower()
    if not filename.endswith(".xlsx"):
        return _safe_json_response({"error": "Only .xlsx files are supported"}, status=400)

    try:
        target_dir = os.path.join(settings.BASE_DIR, "dashboard", "static", "dashboard")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, DASHBOARD2_DEFAULT_XLSX)
        with open(target_path, "wb") as f:
            f.write(upload.read())

        request.session[DASHBOARD2_SESSION_EXCEL_KEY] = target_path
        payload = _dashboard2_prepare_payload(target_path, {})
        payload["uploaded"] = True
        payload["data_source"] = "excel"
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@require_GET
def dashboard2_data(request):
    """
    Reads Marketing dashboard workbook and returns chart payload.
    Query params:
    - sales_type: Primary | Secondary | All
    - brand, sub_brand, category, channel, branch
    - month_from, month_to
    - period: MTD | YTD | All
    """
    try:
        excel_path = _dashboard2_resolve_excel_path(request)
        payload = _dashboard2_prepare_payload(excel_path, request.GET)
        payload["data_source"] = "excel"
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)



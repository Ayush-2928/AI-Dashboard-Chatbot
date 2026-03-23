from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import math
import uuid
import os
import base64
from .utils import (
    execute_dashboard_logic_databricks,
    execute_dashboard_filter_refresh_databricks,
    generate_custom_chart_from_prompt_databricks,
    generate_custom_kpi_from_prompt_databricks,
    fetch_filter_values_databricks,
    generate_wireframe_from_prompt,
    generate_wireframe_artifact_from_prompt,
)
from .cockpit_utils import (
    load_and_prepare_cockpit_excel,
    load_and_prepare_cockpit_databricks,
    fetch_cockpit_table_metadata,
    generate_cockpit_column_mapping_suggestions,
    normalize_cockpit_column_mapping,
    serialize_df_for_session,
    deserialize_df_from_session,
    build_cockpit_payload,
    generate_cockpit_custom_chart_from_prompt,
)
from .models import SavedDashboard


COCKPIT_SESSION_DF_KEY = "cockpit_excel_df_split"
COCKPIT_SESSION_META_KEY = "cockpit_excel_meta"
COCKPIT_SESSION_SOURCE_KEY = "cockpit_data_source"


def _parse_cockpit_column_mapping(raw_value):
    if not raw_value:
        return {}
    parsed = {}
    if isinstance(raw_value, dict):
        parsed = raw_value
    else:
        try:
            parsed = json.loads(str(raw_value))
        except Exception:
            parsed = {}
    if not isinstance(parsed, dict):
        return {}
    return normalize_cockpit_column_mapping(parsed)

def _merge_applied_filters(filters_json, dashboard_data):
    try:
        parsed = json.loads(filters_json) if filters_json else {}
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    if isinstance(dashboard_data, dict):
        selected = str(dashboard_data.get('selected_date_column') or '').strip()
        if selected:
            parsed['_date_column'] = selected

        start = str(dashboard_data.get('applied_start_date') or '').strip()
        end = str(dashboard_data.get('applied_end_date') or '').strip()
        if start:
            parsed['_start_date'] = start
        if end:
            parsed['_end_date'] = end

    return json.dumps(parsed)


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

def index(request):
    """Renders the frontend HTML."""
    return render(request, 'dashboard/index.html')


@csrf_exempt
def dashboard_save(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    payload = _parse_request_payload(request)
    name = str(payload.get("name") or "").strip()
    if not name:
        return _safe_json_response({"error": "name is required"}, status=400)

    dashboard_type = str(payload.get("dashboard_type") or "").strip().lower()
    if dashboard_type not in {"ai_dashboard", "sales_cockpit"}:
        return _safe_json_response({"error": "dashboard_type must be 'ai_dashboard' or 'sales_cockpit'"}, status=400)

    raw_dashboard_payload = payload.get("payload")
    if isinstance(raw_dashboard_payload, (dict, list)):
        dashboard_payload = raw_dashboard_payload
    else:
        raw_payload_text = str(raw_dashboard_payload or "").strip()
        if not raw_payload_text:
            dashboard_payload = {}
        else:
            try:
                dashboard_payload = json.loads(raw_payload_text)
            except Exception:
                return _safe_json_response({"error": "payload must be valid JSON"}, status=400)
            if not isinstance(dashboard_payload, (dict, list)):
                return _safe_json_response({"error": "payload must be a JSON object or array"}, status=400)

    filters_json = _json_text(payload.get("filters_json"), default="{}")
    meta_json = _json_text(payload.get("meta_json"), default="{}")
    thumbnail_hint = str(payload.get("thumbnail_hint") or "").strip()
    if not thumbnail_hint:
        thumbnail_hint = "sales_cockpit" if dashboard_type == "sales_cockpit" else "ai"

    saved_id = str(payload.get("id") or "").strip()
    try:
        if saved_id:
            obj = SavedDashboard.objects.filter(id=saved_id).first()
            if obj is None:
                return _safe_json_response({"error": "Saved dashboard not found"}, status=404)
            obj.name = name
            obj.dashboard_type = dashboard_type
            obj.payload = dashboard_payload
            obj.filters_json = filters_json
            obj.meta_json = meta_json
            obj.thumbnail_hint = thumbnail_hint
            obj.save(update_fields=["name", "dashboard_type", "payload", "filters_json", "meta_json", "thumbnail_hint", "updated_at"])
        else:
            obj = SavedDashboard.objects.create(
                name=name,
                dashboard_type=dashboard_type,
                payload=dashboard_payload,
                filters_json=filters_json,
                meta_json=meta_json,
                thumbnail_hint=thumbnail_hint,
            )
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)

    return _safe_json_response(
        {
            "id": str(obj.id),
            "name": obj.name,
            "saved_at": obj.updated_at.isoformat(),
        }
    )


def dashboard_history(request):
    if request.method != "GET":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    dashboards = SavedDashboard.objects.all()[:50]
    items = []
    for d in dashboards:
        items.append(
            {
                "id": str(d.id),
                "name": d.name,
                "dashboard_type": d.dashboard_type,
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
                "thumbnail_hint": d.thumbnail_hint or "",
            }
        )
    return _safe_json_response({"dashboards": items})


@csrf_exempt
def dashboard_history_detail(request, dashboard_id):
    obj = SavedDashboard.objects.filter(id=dashboard_id).first()
    if obj is None:
        return _safe_json_response({"error": "Saved dashboard not found"}, status=404)

    if request.method == "GET":
        return _safe_json_response(
            {
                "id": str(obj.id),
                "name": obj.name,
                "dashboard_type": obj.dashboard_type,
                "payload": obj.payload,
                "filters_json": obj.filters_json or "{}",
                "meta_json": obj.meta_json or "{}",
                "created_at": obj.created_at.isoformat(),
                "updated_at": obj.updated_at.isoformat(),
                "thumbnail_hint": obj.thumbnail_hint or "",
            }
        )

    if request.method == "DELETE":
        obj.delete()
        return _safe_json_response({"deleted": True})

    if request.method == "PATCH":
        payload = _parse_request_payload(request)
        name = str(payload.get("name") or "").strip()
        if not name:
            return _safe_json_response({"error": "name is required"}, status=400)
        obj.name = name
        obj.save(update_fields=["name", "updated_at"])
        return _safe_json_response(
            {
                "id": str(obj.id),
                "name": obj.name,
                "updated_at": obj.updated_at.isoformat(),
            }
        )

    return _safe_json_response({"error": "Method not allowed"}, status=405)


@csrf_exempt
def process_file(request):
    """Databricks-only dashboard processing endpoint."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    filters_json = request.POST.get('filters', '{}')

    try:
        cached_filter_defs = request.session.get('cached_filter_defs')
        if not isinstance(cached_filter_defs, list):
            cached_filter_defs = None

        last_dashboard_logs = request.session.get('last_dashboard_generation_logs')
        if not isinstance(last_dashboard_logs, list):
            # Backward-compatible fallback for older sessions.
            last_dashboard_logs = request.session.get('last_dashboard_llm_logs')
            if not isinstance(last_dashboard_logs, list):
                last_dashboard_logs = []

        cached_date_range = request.session.get('cached_date_range')
        cached_selected_date_column = str(request.session.get('cached_selected_date_column') or '').strip().lower()

        requested_date_column = ''
        try:
            parsed_filters = json.loads(filters_json) if filters_json else {}
            if isinstance(parsed_filters, dict):
                requested_date_column = str(parsed_filters.get('_date_column') or '').strip().lower()
        except Exception:
            requested_date_column = ''

        date_range_override = None
        if isinstance(cached_date_range, dict) and requested_date_column and requested_date_column == cached_selected_date_column:
            date_range_override = cached_date_range

        session_id = request.session.get('databricks_session_id') or str(uuid.uuid4())
        dashboard_data = execute_dashboard_logic_databricks(
            filters_json,
            session_id=session_id,
            filters_override=cached_filter_defs,
            date_range_override=date_range_override,
        )

        cache_payload = dashboard_data.pop('__cache', {}) if isinstance(dashboard_data, dict) else {}
        if isinstance(cache_payload, dict):
            filters_cache = cache_payload.get('filters')
            if isinstance(filters_cache, list):
                request.session['cached_filter_defs'] = filters_cache

            selected_date_col_cache = cache_payload.get('selected_date_column')
            if selected_date_col_cache:
                request.session['cached_selected_date_column'] = str(selected_date_col_cache)

            date_range_cache = cache_payload.get('date_range')
            if isinstance(date_range_cache, dict):
                request.session['cached_date_range'] = date_range_cache

        current_dashboard_logs = _extract_dashboard_generation_logs(dashboard_data.get('logs', []))
        if current_dashboard_logs:
            request.session['last_dashboard_generation_logs'] = current_dashboard_logs
            request.session['last_dashboard_llm_logs'] = _extract_llm_logs(current_dashboard_logs)
            dashboard_data['logs'] = current_dashboard_logs
        else:
            dashboard_data['logs'] = last_dashboard_logs

        request.session['data_mode'] = 'databricks'
        request.session['active_filters_json'] = _merge_applied_filters(filters_json, dashboard_data)
        request.session['databricks_session_id'] = session_id
        return _safe_json_response(dashboard_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_logs = _extract_dashboard_generation_logs(getattr(e, "_dashboard_logs", []))
        payload = {"error": str(e)}
        if err_logs:
            payload["logs"] = err_logs
        return _safe_json_response(payload, status=500)


@csrf_exempt
def reapply_filters(request):
    """Re-runs only current on-screen widgets with new filters (no LLM regeneration)."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    filters_json = request.POST.get('filters', '{}')
    widgets_json = request.POST.get('widgets', '{}')

    try:
        try:
            widgets_payload = json.loads(widgets_json) if widgets_json else {}
        except Exception:
            widgets_payload = {}
        if not isinstance(widgets_payload, dict):
            widgets_payload = {}

        cached_filter_defs = request.session.get('cached_filter_defs')
        if not isinstance(cached_filter_defs, list):
            cached_filter_defs = None

        cached_date_range = request.session.get('cached_date_range')
        cached_selected_date_column = str(request.session.get('cached_selected_date_column') or '').strip().lower()

        requested_date_column = ''
        try:
            parsed_filters = json.loads(filters_json) if filters_json else {}
            if isinstance(parsed_filters, dict):
                requested_date_column = str(parsed_filters.get('_date_column') or '').strip().lower()
        except Exception:
            requested_date_column = ''

        date_range_override = None
        if isinstance(cached_date_range, dict) and requested_date_column and requested_date_column == cached_selected_date_column:
            date_range_override = cached_date_range

        session_id = request.session.get('databricks_session_id') or str(uuid.uuid4())
        dashboard_data = execute_dashboard_filter_refresh_databricks(
            active_filters_json=filters_json,
            widget_state=widgets_payload,
            session_id=session_id,
            filters_override=cached_filter_defs,
            date_range_override=date_range_override,
        )

        cache_payload = dashboard_data.pop('__cache', {}) if isinstance(dashboard_data, dict) else {}
        if isinstance(cache_payload, dict):
            filters_cache = cache_payload.get('filters')
            if isinstance(filters_cache, list):
                request.session['cached_filter_defs'] = filters_cache

            selected_date_col_cache = cache_payload.get('selected_date_column')
            if selected_date_col_cache:
                request.session['cached_selected_date_column'] = str(selected_date_col_cache)

            date_range_cache = cache_payload.get('date_range')
            if isinstance(date_range_cache, dict):
                request.session['cached_date_range'] = date_range_cache

        request.session['active_filters_json'] = _merge_applied_filters(filters_json, dashboard_data)
        request.session['databricks_session_id'] = session_id

        # Reapply path has no LLM generation; surface operational logs for debugging/perf.
        refresh_logs = _extract_filter_refresh_logs(dashboard_data.get('logs', []))
        dashboard_data['logs'] = refresh_logs if refresh_logs else []

        return _safe_json_response(dashboard_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def custom_chart(request):
    """Generates a chart or KPI from natural language using Databricks mode."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    prompt = (request.POST.get('prompt') or '').strip()
    if not prompt:
        return _safe_json_response({"error": "Prompt is required"}, status=400)

    artifact_type = (request.POST.get('artifact_type') or 'chart').strip().lower()
    if artifact_type not in {'chart', 'kpi'}:
        artifact_type = 'chart'

    try:
        filters_json = request.POST.get('filters')
        if not filters_json:
            filters_json = request.session.get('active_filters_json', '{}')
        clarification_choice = request.POST.get('clarification_choice')
        clarification_payload = None
        if clarification_choice:
            try:
                import json
                clarification_payload = json.loads(clarification_choice)
            except Exception:
                clarification_payload = None

        force_ambiguity_fallback_raw = (request.POST.get('force_ambiguity_fallback') or '').strip().lower()
        force_ambiguity_fallback = force_ambiguity_fallback_raw in {'1', 'true', 'yes', 'on'}

        if artifact_type == 'kpi':
            result = generate_custom_kpi_from_prompt_databricks(
                prompt,
                active_filters_json=filters_json,
                clarification_choice=clarification_payload,
                allow_ambiguity_fallback=force_ambiguity_fallback,
            )
        else:
            result = generate_custom_chart_from_prompt_databricks(
                prompt,
                active_filters_json=filters_json,
                clarification_choice=clarification_payload,
                allow_ambiguity_fallback=force_ambiguity_fallback,
            )

        llm_logs = _extract_llm_logs(result.get('logs', []))
        if llm_logs:
            result['logs'] = llm_logs
        else:
            result['logs'] = []
        return _safe_json_response(result)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def filter_values(request):
    """Fetches filter values for one selected filter column (lazy loading)."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    column = (request.POST.get('column') or '').strip()
    if not column:
        return _safe_json_response({"error": "Column is required"}, status=400)

    filters_json = request.POST.get('filters')
    if not filters_json:
        filters_json = request.session.get('active_filters_json', '{}')

    try:
        result = fetch_filter_values_databricks(column_name=column, active_filters_json=filters_json)
        return _safe_json_response(result)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


def wireframe(request):
    """Renders the Wireframe Maker page."""
    return render(request, 'dashboard/wireframe.html')


def sales_cockpit(request):
    """Renders standalone Sales Cockpit page (supports Excel or Databricks)."""
    return render(request, 'dashboard/sales_cockpit.html')


@csrf_exempt
def sales_cockpit_metadata(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    invoice_table = str(request.POST.get("invoice_table") or "llm_test.llm.invoice_template").strip()
    target_table = str(request.POST.get("target_table") or "llm_test.llm.target_template").strip()
    try:
        metadata = fetch_cockpit_table_metadata(
            invoice_table=invoice_table,
            target_table=target_table,
        )
        return _safe_json_response(metadata)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def sales_cockpit_column_mapping(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    invoice_table = str(request.POST.get("invoice_table") or "llm_test.llm.invoice_template").strip()
    target_table = str(request.POST.get("target_table") or "llm_test.llm.target_template").strip()

    metadata_raw = request.POST.get("metadata") or ""
    metadata = {}
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    invoice_columns = metadata.get("invoice_columns") if isinstance(metadata, dict) else None
    target_columns = metadata.get("target_columns") if isinstance(metadata, dict) else None
    if not isinstance(invoice_columns, list) or not isinstance(target_columns, list):
        try:
            metadata = fetch_cockpit_table_metadata(
                invoice_table=invoice_table,
                target_table=target_table,
            )
            invoice_columns = metadata.get("invoice_columns") or []
            target_columns = metadata.get("target_columns") or []
        except Exception as e:
            return _safe_json_response({"error": str(e)}, status=500)

    try:
        mapping_result = generate_cockpit_column_mapping_suggestions(
            invoice_columns=invoice_columns or [],
            target_columns=target_columns or [],
            invoice_table=invoice_table,
            target_table=target_table,
        )
        return _safe_json_response(
            {
                "invoice_table": invoice_table,
                "target_table": target_table,
                "mapping": mapping_result.get("mapping", {}),
                "suggestions": mapping_result.get("suggestions", {}),
                "model": mapping_result.get("model", "gpt-4o"),
                "logs": _extract_llm_logs(mapping_result.get("logs", [])),
            }
        )
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


def _get_cockpit_df_from_session(request):
    raw = request.session.get(COCKPIT_SESSION_DF_KEY)
    return deserialize_df_from_session(raw)


@csrf_exempt
def sales_cockpit_upload(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    upload = request.FILES.get("file")
    if not upload:
        return _safe_json_response({"error": "Excel file is required"}, status=400)

    filename = str(getattr(upload, "name", "") or "").lower()
    if not filename.endswith(".xlsx"):
        return _safe_json_response({"error": "Only .xlsx files are supported"}, status=400)

    try:
        blob = upload.read()
        df, meta = load_and_prepare_cockpit_excel(blob)
        request.session[COCKPIT_SESSION_DF_KEY] = serialize_df_for_session(df)
        request.session[COCKPIT_SESSION_META_KEY] = meta
        request.session[COCKPIT_SESSION_SOURCE_KEY] = "excel"
        payload = build_cockpit_payload(df, filters={})
        payload["meta"] = meta
        payload["uploaded"] = True
        payload["data_source"] = "excel"
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def sales_cockpit_databricks_init(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    invoice_table = str(request.POST.get("invoice_table") or "llm_test.llm.invoice_template").strip()
    target_table = str(request.POST.get("target_table") or "llm_test.llm.target_template").strip()
    column_mapping = _parse_cockpit_column_mapping(request.POST.get("column_mapping"))

    try:
        _, meta = load_and_prepare_cockpit_databricks(
            invoice_table=invoice_table,
            target_table=target_table,
            column_mapping=column_mapping,
        )
        request.session.pop(COCKPIT_SESSION_DF_KEY, None)
        request.session[COCKPIT_SESSION_META_KEY] = {
            "source_mode": "databricks",
            "invoice_table": invoice_table,
            "target_table": target_table,
            "filter_options": (meta or {}).get("filter_options", {}),
            "selected_month": (meta or {}).get("month_key", ""),
            "column_mapping": (meta or {}).get("column_mapping") or column_mapping,
        }
        request.session[COCKPIT_SESSION_SOURCE_KEY] = "databricks"
        payload = build_cockpit_payload(
            None,
            filters={},
            data_source="databricks",
            databricks_config=request.session.get(COCKPIT_SESSION_META_KEY, {}),
            invoice_table=invoice_table,
            target_table=target_table,
            selected_month=(meta or {}).get("month_key", ""),
            column_mapping=(meta or {}).get("column_mapping") or column_mapping,
        )
        payload["meta"] = request.session.get(COCKPIT_SESSION_META_KEY, {})
        payload["uploaded"] = True
        payload["data_source"] = "databricks"
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def sales_cockpit_data(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    source = str(request.session.get(COCKPIT_SESSION_SOURCE_KEY) or "excel").strip().lower()
    meta = request.session.get(COCKPIT_SESSION_META_KEY, {})

    raw_filters = request.POST.get("filters") or "{}"
    try:
        filters = json.loads(raw_filters) if raw_filters else {}
    except Exception:
        filters = {}
    if not isinstance(filters, dict):
        filters = {}

    try:
        if source == "databricks":
            selected_month = str(filters.get("selected_month") or meta.get("selected_month") or "").strip()
            if selected_month:
                meta["selected_month"] = selected_month
                request.session[COCKPIT_SESSION_META_KEY] = meta
            payload = build_cockpit_payload(
                None,
                filters=filters,
                data_source="databricks",
                databricks_config=meta,
                invoice_table=meta.get("invoice_table"),
                target_table=meta.get("target_table"),
                selected_month=selected_month,
                column_mapping=meta.get("column_mapping"),
            )
        else:
            df = _get_cockpit_df_from_session(request)
            if df is None or df.empty:
                return _safe_json_response({"error": "No cockpit dataset found. Upload Excel or choose Databricks first."}, status=400)
            payload = build_cockpit_payload(df, filters=filters, data_source="excel")
        payload["meta"] = meta
        payload["uploaded"] = True
        payload["data_source"] = source or "excel"
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def sales_cockpit_custom_chart(request):
    if request.method != "POST":
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    prompt = str(request.POST.get("prompt") or "").strip()
    if not prompt:
        return _safe_json_response({"error": "Prompt is required"}, status=400)

    source = str(request.session.get(COCKPIT_SESSION_SOURCE_KEY) or "excel").strip().lower()
    meta = request.session.get(COCKPIT_SESSION_META_KEY, {}) if isinstance(request.session.get(COCKPIT_SESSION_META_KEY, {}), dict) else {}

    df = None
    if source != "databricks":
        df = _get_cockpit_df_from_session(request)
        if df is None or df.empty:
            return _safe_json_response({"error": "No cockpit dataset found. Upload Excel or choose Databricks first."}, status=400)

    raw_filters = request.POST.get("filters") or "{}"
    try:
        filters = json.loads(raw_filters) if raw_filters else {}
    except Exception:
        filters = {}
    if not isinstance(filters, dict):
        filters = {}

    existing_chart_raw = request.POST.get("existing_chart") or ""
    existing_chart = None
    if existing_chart_raw:
        try:
            parsed = json.loads(existing_chart_raw)
            if isinstance(parsed, dict):
                existing_chart = parsed
        except Exception:
            existing_chart = None

    try:
        selected_month = str(filters.get("selected_month") or meta.get("selected_month") or "").strip()
        result = generate_cockpit_custom_chart_from_prompt(
            df=df,
            user_prompt=prompt,
            active_filters=filters,
            existing_chart=existing_chart,
            invoice_table=meta.get("invoice_table") or "llm_test.llm.invoice_template",
            target_table=meta.get("target_table") or "llm_test.llm.target_template",
            selected_month=selected_month,
            data_source=source,
            column_mapping=meta.get("column_mapping"),
        )
        raw_logs = result.get("logs", []) if isinstance(result, dict) else []
        if isinstance(raw_logs, list):
            result["logs"] = _extract_llm_logs(raw_logs) or [str(x) for x in raw_logs][-15:]
        return _safe_json_response(result)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def wireframe_generate(request):
    """Generate KPI wireframe blueprint and synthetic sample payload from plain-language requirements."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    prompt = str(request.POST.get("description") or "").strip()
    if not prompt:
        return _safe_json_response({"error": "Description is required"}, status=400)

    requested_kpis_raw = str(request.POST.get("kpi_count") or "6").strip()
    try:
        requested_kpis = max(3, min(8, int(requested_kpis_raw)))
    except Exception:
        requested_kpis = 6

    requested_charts_raw = str(request.POST.get("chart_count") or "3").strip()
    try:
        requested_charts = max(0, min(8, int(requested_charts_raw)))
    except Exception:
        requested_charts = 3

    image_file = request.FILES.get("reference_image")
    image_b64 = ""
    image_mime = "image/png"

    if image_file:
        try:
            max_bytes = 4 * 1024 * 1024
            blob = image_file.read(max_bytes + 1)
            if len(blob) > max_bytes:
                return _safe_json_response({"error": "Reference image must be <= 4MB"}, status=400)
            mime = str(getattr(image_file, "content_type", "") or "image/png").strip()
            if not mime.startswith("image/"):
                mime = "image/png"
            image_mime = mime
            image_b64 = base64.b64encode(blob).decode("utf-8")
        except Exception as e:
            return _safe_json_response({"error": f"Failed to read reference image: {str(e)}"}, status=400)

    try:
        payload = generate_wireframe_from_prompt(
            description=prompt,
            kpi_count=requested_kpis,
            chart_count=requested_charts,
            reference_image_b64=image_b64,
            reference_mime=image_mime,
        )
        raw_logs = payload.get("logs", []) if isinstance(payload, dict) else []
        if isinstance(raw_logs, list):
            llm_logs = _extract_llm_logs(raw_logs)
            wire_logs = [str(x) for x in raw_logs if str(x).startswith("[WIRE]")]
            payload["logs"] = (llm_logs + wire_logs)[-20:] or [str(x) for x in raw_logs][-12:]
        return _safe_json_response(payload)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


@csrf_exempt
def wireframe_add_artifact(request):
    """Adds one KPI/chart to an existing wireframe state using LLM intent."""
    if request.method != 'POST':
        return _safe_json_response({"error": "Method not allowed"}, status=405)

    prompt = str(request.POST.get("prompt") or "").strip()
    if not prompt:
        return _safe_json_response({"error": "Prompt is required"}, status=400)

    artifact_hint = str(request.POST.get("artifact_hint") or "auto").strip().lower()
    if artifact_hint not in {"auto", "kpi", "chart"}:
        artifact_hint = "auto"

    raw_state = request.POST.get("current_state") or "{}"
    try:
        current_state = json.loads(raw_state) if raw_state else {}
    except Exception:
        current_state = {}
    if not isinstance(current_state, dict):
        current_state = {}

    try:
        result = generate_wireframe_artifact_from_prompt(
            user_prompt=prompt,
            current_payload=current_state,
            artifact_hint=artifact_hint,
        )
        raw_logs = result.get("logs", []) if isinstance(result, dict) else []
        if isinstance(raw_logs, list):
            llm_logs = _extract_llm_logs(raw_logs)
            result["logs"] = llm_logs or [str(x) for x in raw_logs][-12:]
        return _safe_json_response(result)
    except Exception as e:
        return _safe_json_response({"error": str(e)}, status=500)


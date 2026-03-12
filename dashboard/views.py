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

def index(request):
    """Renders the frontend HTML."""
    return render(request, 'dashboard/index.html')


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


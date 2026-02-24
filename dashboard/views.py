from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import uuid
from .utils import (
    execute_dashboard_logic_databricks,
    execute_dashboard_filter_refresh_databricks,
    generate_custom_chart_from_prompt_databricks,
    generate_custom_kpi_from_prompt_databricks,
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
    keep = []
    for line in lines:
        s = str(line or '')
        if ('[LLM REQUEST]' in s) or ('[LLM RESPONSE]' in s) or ('[LLM ERROR]' in s):
            keep.append(s)
    return keep

def index(request):
    """Renders the frontend HTML."""
    return render(request, 'dashboard/index.html')


@csrf_exempt
def process_file(request):
    """Databricks-only dashboard processing endpoint."""
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    filters_json = request.POST.get('filters', '{}')

    try:
        cached_filter_defs = request.session.get('cached_filter_defs')
        if not isinstance(cached_filter_defs, list):
            cached_filter_defs = None

        last_llm_logs = request.session.get('last_dashboard_llm_logs')
        if not isinstance(last_llm_logs, list):
            last_llm_logs = []

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

        current_llm_logs = _extract_llm_logs(dashboard_data.get('logs', []))
        if current_llm_logs:
            request.session['last_dashboard_llm_logs'] = current_llm_logs
            dashboard_data['logs'] = current_llm_logs
        else:
            dashboard_data['logs'] = last_llm_logs

        request.session['data_mode'] = 'databricks'
        request.session['active_filters_json'] = _merge_applied_filters(filters_json, dashboard_data)
        request.session['databricks_session_id'] = session_id
        return JsonResponse(dashboard_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def reapply_filters(request):
    """Re-runs only current on-screen widgets with new filters (no LLM regeneration)."""
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

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

        # Keep frontend logs clean for filter-only refresh path.
        llm_logs = _extract_llm_logs(dashboard_data.get('logs', []))
        dashboard_data['logs'] = llm_logs if llm_logs else []

        return JsonResponse(dashboard_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def custom_chart(request):
    """Generates a chart or KPI from natural language using Databricks mode."""
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    prompt = (request.POST.get('prompt') or '').strip()
    if not prompt:
        return JsonResponse({"error": "Prompt is required"}, status=400)

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
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



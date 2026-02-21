from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import uuid
from .utils import (
    execute_dashboard_logic_databricks,
    generate_custom_chart_from_prompt_databricks,
    generate_custom_kpi_from_prompt_databricks,
)


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
        session_id = str(uuid.uuid4())
        dashboard_data = execute_dashboard_logic_databricks(filters_json, session_id=session_id)
        request.session['data_mode'] = 'databricks'
        request.session['active_filters_json'] = filters_json
        request.session['databricks_session_id'] = session_id
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
        filters_json = request.session.get('active_filters_json', '{}')
        clarification_choice = request.POST.get('clarification_choice')
        clarification_payload = None
        if clarification_choice:
            try:
                import json
                clarification_payload = json.loads(clarification_choice)
            except Exception:
                clarification_payload = None

        if artifact_type == 'kpi':
            result = generate_custom_kpi_from_prompt_databricks(prompt, active_filters_json=filters_json)
        else:
            result = generate_custom_chart_from_prompt_databricks(
                prompt,
                active_filters_json=filters_json,
                clarification_choice=clarification_payload,
            )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

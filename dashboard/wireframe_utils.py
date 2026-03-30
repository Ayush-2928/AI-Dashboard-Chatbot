import json
import base64
from django.views.decorators.csrf import csrf_exempt
from .utils import _safe_json_response, _extract_llm_logs, generate_wireframe_from_prompt, generate_wireframe_artifact_from_prompt

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



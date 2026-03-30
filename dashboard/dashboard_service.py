import json
from django.views.decorators.csrf import csrf_exempt
from .models import SavedDashboard
from .utils import _safe_json_response, _parse_request_payload, _json_text

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



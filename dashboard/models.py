import uuid

from django.db import models


class SavedDashboard(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    dashboard_type = models.CharField(max_length=50)  # ai_dashboard | sales_cockpit
    payload = models.JSONField()
    filters_json = models.TextField(default="{}")
    meta_json = models.TextField(default="{}")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    thumbnail_hint = models.CharField(max_length=100, default="")

    class Meta:
        ordering = ["-updated_at"]

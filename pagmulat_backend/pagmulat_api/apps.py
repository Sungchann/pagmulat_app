# pagmulat_api/apps.py
from django.apps import AppConfig
from django.core.cache import cache

class PagmulatApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pagmulat_api"
    
    def ready(self):
        # Clear ARM cache on startup
        cache.delete('arm_dashboard_data')
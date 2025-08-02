# pagmulat_api/tasks.py
from celery import shared_task
from .train_model import unified_training_pipeline
from django.conf import settings
import os

@shared_task
def run_training_pipeline():
    """Celery task for training pipeline"""
    config = {
        'data_path': os.path.join(settings.BASE_DIR, "ModifiedFinalData.xlsx"),
        'target_column': "Productive_Yes",
        'min_support': 0.15,
        'min_confidence': 0.65,
        'min_lift': 1.2
    }
    return unified_training_pipeline(**config)
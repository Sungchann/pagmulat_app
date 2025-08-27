# pagmulat_api/urls.py
from django.urls import path
from . import views
from .views_api.pipeline_data_views import get_raw_data, get_cleaned_data, get_augmented_data
from .views_api.rules_comparison_views import get_algorithm_rules
from .assessment_api.assessment_pipeline import assessment_predict

urlpatterns = [
    # ... existing endpoints ...
    path('dashboard/', views.arm_dashboard, name='arm-dashboard'),
    path('arm/dashboard/', views.arm_dashboard, name='arm-dashboard-alt'), # Alternative URL for frontend compatibility
    # ARM specific endpoints
    path('arm/metadata/', views.arm_metadata, name='arm-metadata'),
    path('arm/rules/', views.arm_rules, name='arm-rules'),
    path('patterns/<str:behavior>/', views.behavior_patterns, name='behavior-patterns'),
    path('predict/', views.predict, name='predict'),
    path('train/', views.train, name='train-model'),
    path('all_students/', views.all_students, name='all-students'),
    path('prediction_history/', views.prediction_history, name='prediction-history'),
    path('arm-rules/', views.get_all_rules, name='all-rules'),
    path('frequent-itemsets/', views.get_all_itemsets, name='all-itemsets'),
    path('prediction-history/', views.prediction_history, name='prediction-history'),
    path('assessment-predict/', assessment_predict, name='assessment_predict'),
    
    # Data Obtention APIs
    path('raw-data/', get_raw_data, name='raw-data'),
    path('cleaned-data/', get_cleaned_data, name='cleaned-data'),
    path('augmented-data/', get_augmented_data, name='augmented-data'),
    
    # Rules Obtention API
    path('apriori-fpgrowth-rules/', get_algorithm_rules, name="algorithm-rules")
]
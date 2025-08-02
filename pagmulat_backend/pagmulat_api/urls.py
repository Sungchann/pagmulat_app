# pagmulat_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # ... existing endpoints ...
    path('dashboard/', views.arm_dashboard, name='arm-dashboard'),
    path('patterns/<str:behavior>/', views.behavior_patterns, name='behavior-patterns'),
    path('predict/', views.predict, name='predict'),
    path('train/', views.train, name='train-model'),
    path('all_students/', views.all_students, name='all-students'),
    path('prediction_history/', views.prediction_history, name='prediction-history'),
    path('arm-rules/', views.get_all_rules, name='all-rules'),
    path('frequent-itemsets/', views.get_all_itemsets, name='all-itemsets'),
    path('prediction-history/', views.prediction_history, name='prediction-history'),
]
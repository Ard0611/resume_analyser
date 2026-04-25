from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_resume, name='api_upload'),
    path('analyze/', views.analyze_resume, name='api_analyze'),
    path('match-jd/', views.match_jd, name='api_match_jd'),
    path('history/', views.history, name='api_history'),
]

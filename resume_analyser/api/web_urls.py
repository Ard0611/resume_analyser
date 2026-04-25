from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_page, name='upload'),
    path('result/<int:resume_id>/', views.result_page, name='result'),
    path('history/', views.history_page, name='history'),
]

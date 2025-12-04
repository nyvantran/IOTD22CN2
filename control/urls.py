from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/command/', views.get_command, name='get_command'),
    path('api/set-command/', views.set_command, name='set_command'),
    path('api/history/', views.command_history, name='command_history'),
    path('api/analyze-stream/', views.analyze_stream_once, name='analyze_stream'),
    path('api/detect/', views.detect_uploaded_image, name='detect_uploaded_image'),
    path('api/get-analysis/', views.get_analysis_result, name='get_analysis'),
    path('api/stream-live-feed/', views.stream_live_feed, name='stream_live_feed'),

]

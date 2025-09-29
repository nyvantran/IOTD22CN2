from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/command/', views.get_command, name='get_command'),
    path('api/set-command/', views.set_command, name='set_command'),
    path('api/history/', views.command_history, name='command_history'),
]
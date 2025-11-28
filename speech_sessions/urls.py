from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .api import SpeechSessionViewSet

# Create a router for the API endpoints
router = DefaultRouter()
router.register(r'sessions', SpeechSessionViewSet, basename='session')

app_name = 'speech_sessions'

# Web interface URLs
urlpatterns = [
    # Main session management pages
    path('', views.session_list_view, name='session_list'),
    path('create/', views.session_create_view, name='session_create'),
    path('<int:pk>/', views.session_detail_view, name='session_detail'),
    path('<int:pk>/update/', views.session_update_view, name='session_update'),
    path('<int:pk>/delete/', views.session_delete_view, name='session_delete'),
    
    # Additional functionality
    path('analytics/', views.session_analytics_view, name='session_analytics'),
    path('bulk-action/', views.session_bulk_action_view, name='session_bulk_action'),
    
    # API endpoints - mounted under /api/
    path('api/', include(router.urls)),
]


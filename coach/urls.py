from django.urls import path, include
from . import views

app_name = 'coach'

urlpatterns = [
    # Authentication URLs
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Email Verification URLs
    path('verification-sent/', views.verification_sent_view, name='verification_sent'),
    path('verify-email/<str:token>/', views.verify_email_view, name='verify_email'),
    path('email-verified/', views.email_verified_view, name='email_verified'),
    path('resend-verification/', views.resend_verification_view, name='resend_verification'),
    
    # Password Reset URLs
    path('password-reset/', views.CustomPasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', views.CustomPasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', views.CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', views.CustomPasswordResetCompleteView.as_view(), name='password_reset_complete'),
    
    # 2FA URLs
    path('security-settings/', views.security_settings_view, name='security_settings'),
    path('setup-2fa/', views.setup_2fa_view, name='setup_2fa'),
    path('verify-2fa-setup/', views.verify_2fa_setup_view, name='verify_2fa_setup'),
    path('choose-2fa-method/', views.choose_2fa_method_view, name='choose_2fa_method'),
    path('verify-2fa/', views.verify_2fa_view, name='verify_2fa'),
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),
    path('send-fallback-code/', views.send_fallback_code_view, name='send_fallback_code'),
    path('verify-fallback-code/', views.verify_fallback_code_view, name='verify_fallback_code'),
    
    # Main application URLs
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('feedback/<int:feedback_id>/', views.feedback_view, name='feedback_detail'),
    path('profile/', views.profile_view, name='profile'),
    
    # API URLs
    path('api/stats/', views.api_user_stats, name='api_stats'),
]



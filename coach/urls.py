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
    
    # Passkey (WebAuthn) URLs
    path('passkeys/', views.manage_passkeys_view, name='manage_passkeys'),
    path('passkeys/delete/<int:passkey_id>/', views.delete_passkey_view, name='delete_passkey'),
    path('passkeys/register/begin/', views.passkey_registration_begin, name='passkey_registration_begin'),
    path('passkeys/register/complete/', views.passkey_registration_complete, name='passkey_registration_complete'),
    path('passkeys/authenticate/begin/', views.passkey_authentication_begin, name='passkey_authentication_begin'),
    path('passkeys/authenticate/complete/', views.passkey_authentication_complete, name='passkey_authentication_complete'),
    path('verify-passkey/', views.verify_passkey_view, name='verify_passkey'),
    
    # Main application URLs
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('record/', views.record_view, name='record'),
    path('reports/', views.reports_view, name='reports'),
    path('reports/export/<str:format>/', views.export_feedback_report, name='export_feedback_report'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('feedback/<int:feedback_id>/', views.feedback_view, name='feedback_detail'),
    path('profile/', views.profile_view, name='profile'),
    
    # API URLs
    path('api/stats/', views.api_user_stats, name='api_stats'),
    
    # Drill URLs
    path('drills/', views.drills_list_view, name='drills_list'),
    path('drill/<int:pk>/', views.DrillDetailView.as_view(), name='drill_detail'),
    
    # Pronunciation Drills
    path('drill/poem-echo/', views.PoemEchoChallengeView.as_view(), name='drill_poem_echo'),
    path('drill/vowel-vortex/', views.ExaggeratedVowelVortexView.as_view(), name='drill_vowel_vortex'),
    path('drill/mirror-mimic/', views.MirrorMimicMadnessView.as_view(), name='drill_mirror_mimic'),
    path('drill/phonetic-puzzle/', views.PhoneticPuzzleBuilderView.as_view(), name='drill_phonetic_puzzle'),
    path('drill/shadow-superhero/', views.ShadowSuperheroView.as_view(), name='drill_shadow_superhero'),
    path('drill/pencil-precision/', views.PencilPrecisionDrillView.as_view(), name='drill_pencil_precision'),
    
    # Pacing Drills
    path('drill/metronome-rhythm/', views.MetronomeRhythmRaceView.as_view(), name='drill_metronome_rhythm'),
    path('drill/pause-pyramid/', views.PausePyramidBuilderView.as_view(), name='drill_pause_pyramid'),
    path('drill/slow-motion-story/', views.SlowMotionStorySlamView.as_view(), name='drill_slow_motion_story'),
    path('drill/beat-drop-dialogue/', views.BeatDropDialogueView.as_view(), name='drill_beat_drop_dialogue'),
    path('drill/timer-tag-team/', views.TimerTagTeamView.as_view(), name='drill_timer_tag_team'),
    path('drill/echo-chamber/', views.EchoChamberEscalationView.as_view(), name='drill_echo_chamber'),
    
    # Filler Words Drills
    path('drill/filler-zap-game/', views.FillerZapGameView.as_view(), name='drill_filler_zap_game'),
    path('drill/silent-switcheroo/', views.SilentSwitcherooView.as_view(), name='drill_silent_switcheroo'),
    path('drill/pause-powerup/', views.PausePowerUpView.as_view(), name='drill_pause_powerup'),
    path('drill/filler-hunt-mirror/', views.FillerHuntMirrorMazeView.as_view(), name='drill_filler_hunt_mirror'),
    path('drill/word-swap-whirlwind/', views.WordSwapWhirlwindView.as_view(), name='drill_word_swap_whirlwind'),
    path('drill/echo-elimination/', views.EchoEliminationEchoView.as_view(), name='drill_echo_elimination'),
    
    path('complete-drill/', views.complete_drill, name='complete_drill'),
    path('drill-progress-api/', views.drill_progress_api, name='drill_progress_api'),
    
    # Real-time Speech Analysis APIs
    path('api/analyze-speech/', views.analyze_speech_api, name='analyze_speech_api'),
    path('api/analyze-speech/async/', views.analyze_speech_async_upload, name='analyze_speech_async'),
    path('api/analyze-speech/status/<str:task_id>/', views.analyze_speech_task_status, name='analyze_speech_task_status'),
    path('api/detect-fillers/', views.detect_fillers_api, name='detect_fillers_api'),
    path('analyze-drill-api/', views.analyze_drill_api, name='analyze_drill_api'),
]



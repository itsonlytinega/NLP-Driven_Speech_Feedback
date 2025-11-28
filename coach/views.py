from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db import models
from django.conf import settings
from django.core.mail import send_mail
from django.core.files.storage import default_storage
from django.urls import reverse_lazy, reverse
from django.utils import timezone
from django.views.generic import DetailView, TemplateView
from datetime import timedelta
from celery.result import AsyncResult
from django_otp.plugins.otp_totp.models import TOTPDevice
from django_otp.decorators import otp_required
from django_otp import login as otp_login
import qrcode
import qrcode.image.svg
import io
import base64
import spacy
import random
from .forms import CustomUserCreationForm, CustomAuthenticationForm, CustomPasswordResetForm, TwoFactorChoiceForm, OTPVerificationForm
from .models import User, Feedback, Drill, DrillCompletion, EmailOTP, WebAuthnCredential
from .utils import (
    send_verification_email, send_otp_email, create_email_otp, verify_email_token, 
    send_password_reset_email, detect_filler_words, analyze_speech_quality, 
    get_drill_recommendations_by_analysis, transcribe_audio, get_bert_scores,
    analyze_speech_pipeline, DrillRecommender, predict_cause
)
from .tasks import analyze_speech_async
import json
from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
    options_to_json,
)
from webauthn.helpers.structs import (
    PublicKeyCredentialDescriptor,
    UserVerificationRequirement,
    AuthenticatorSelectionCriteria,
    ResidentKeyRequirement,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
import logging

# Setup logging
logger = logging.getLogger(__name__)


def get_spacy_model():
    """Safely load spaCy model with fallback."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to basic English model if en_core_web_sm is not available
        try:
            return spacy.load("en")
        except OSError:
            return None


def signup_view(request):
    """Handle user registration with email verification."""
    if request.user.is_authenticated:
        return redirect('coach:dashboard')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  # Deactivate until email is verified
            user.is_email_verified = False
            user.save()
            
            # Send verification email
            if send_verification_email(user):
                messages.success(request, 'Account created successfully! Please check your email and click the verification link to activate your account.')
                return redirect('coach:verification_sent')
            else:
                messages.error(request, 'Account created but failed to send verification email. Please contact support.')
                return redirect('coach:signup')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'coach/signup.html', {'form': form})


def login_view(request):
    """Handle user login with email verification and 2FA."""
    if request.user.is_authenticated:
        return redirect('coach:dashboard')
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data['user']
            
            # Check if email is verified
            if not user.is_email_verified:
                messages.warning(request, 'Please verify your email address before logging in. Check your email for a verification link.')
                return redirect('coach:resend_verification')
            
            # Store user in session for 2FA process
            request.session['temp_user_id'] = user.id
            
            # Check if user has 2FA enabled
            if user.is_2fa_enabled and user.has_2fa_setup():
                messages.info(request, 'Please complete 2FA verification to continue.')
                return redirect('coach:choose_2fa_method')
            else:
                login(request, user)
                messages.success(request, f'Welcome back, {user.first_name}!')
                return redirect('coach:dashboard')
    else:
        form = CustomAuthenticationForm()
    
    return render(request, 'coach/login.html', {'form': form})


@login_required
def logout_view(request):
    """Handle user logout."""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('coach:login')


@login_required
def dashboard_view(request):
    """Display enhanced user dashboard with feedback, sessions, drills, and analytics."""
    from datetime import timedelta, datetime
    from django.db.models import Avg, Count, Q
    from speech_sessions.models import SpeechSession
    
    # Get all data sources
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    sessions = SpeechSession.objects.filter(user=request.user, status='analyzed').order_by('-date')
    drill_completions = DrillCompletion.objects.filter(user=request.user).order_by('-completed_at')
    
    # Combine all feedback/sessions for timeline (last 30 days)
    thirty_days_ago = timezone.now() - timedelta(days=30)
    
    # Get sessions with scores
    sessions_with_scores = sessions.filter(filler_score__isnull=False)
    
    # Get all feedback with scores (new model scores)
    recent_feedbacks = feedbacks.filter(created_at__gte=thirty_days_ago, filler_score__isnull=False)
    
    # Get latest scores from most recent session (Option 2: Latest scores)
    latest_session = sessions_with_scores.first()
    latest_filler_score = latest_session.filler_score if latest_session and latest_session.filler_score is not None else None
    latest_clarity_score = latest_session.clarity_score if latest_session and latest_session.clarity_score is not None else None
    latest_pacing_score = latest_session.pacing_score if latest_session and latest_session.pacing_score is not None else None
    
    # If no session scores, try to get from latest feedback
    if latest_filler_score is None:
        latest_feedback = feedbacks.filter(filler_score__isnull=False).first()
        if latest_feedback:
            latest_filler_score = latest_feedback.filler_score
            latest_clarity_score = latest_feedback.clarity_score
            latest_pacing_score = latest_feedback.pacing_score
    
    # Calculate average metrics from BERT ensemble scores (Option 2: Average scores)
    # Combine data from both Feedback and SpeechSession
    
    # Get averages from Feedback
    feedback_avg_filler = feedbacks.filter(filler_score__isnull=False).aggregate(avg=models.Avg('filler_score'))['avg']
    feedback_avg_clarity = feedbacks.filter(clarity_score__isnull=False).aggregate(avg=models.Avg('clarity_score'))['avg']
    feedback_avg_pacing = feedbacks.filter(pacing_score__isnull=False).aggregate(avg=models.Avg('pacing_score'))['avg']
    
    # Get averages from SpeechSession
    session_avg_filler = sessions_with_scores.aggregate(avg=models.Avg('filler_score'))['avg']
    session_avg_clarity = sessions_with_scores.aggregate(avg=models.Avg('clarity_score'))['avg']
    session_avg_pacing = sessions_with_scores.aggregate(avg=models.Avg('pacing_score'))['avg']
    
    # Combine averages (weighted by count)
    feedback_count = feedbacks.filter(filler_score__isnull=False).count()
    session_count = sessions_with_scores.count()
    total_count = feedback_count + session_count
    
    if total_count > 0:
        avg_filler_score = ((feedback_avg_filler or 0) * feedback_count + (session_avg_filler or 0) * session_count) / total_count if total_count > 0 else 0
        avg_clarity_score = ((feedback_avg_clarity or 0) * feedback_count + (session_avg_clarity or 0) * session_count) / total_count if total_count > 0 else 0
        avg_pacing_score = ((feedback_avg_pacing or 0) * feedback_count + (session_avg_pacing or 0) * session_count) / total_count if total_count > 0 else 0
    else:
        avg_filler_score = feedback_avg_filler or session_avg_filler or 0
        avg_clarity_score = feedback_avg_clarity or session_avg_clarity or 0
        avg_pacing_score = feedback_avg_pacing or session_avg_pacing or 0
    
    # Legacy metrics (fallback)
    avg_f1_score = feedbacks.aggregate(avg=models.Avg('f1_score'))['avg'] or 0
    avg_fillers = feedbacks.aggregate(avg=models.Avg('fillers_count'))['avg'] or 0
    avg_wpm = feedbacks.aggregate(avg=models.Avg('wpm'))['avg'] or 0
    
    # Calculate improvement trends (last 7 days vs previous 7 days)
    now = timezone.now()
    last_7_days = now - timedelta(days=7)
    last_14_days = now - timedelta(days=14)
    
    # Get recent data from both Feedback and SpeechSession
    recent_7_feedback = feedbacks.filter(created_at__gte=last_7_days, filler_score__isnull=False)
    previous_7_feedback = feedbacks.filter(created_at__gte=last_14_days, created_at__lt=last_7_days, filler_score__isnull=False)
    recent_7_sessions = sessions_with_scores.filter(date__gte=last_7_days)
    previous_7_sessions = sessions_with_scores.filter(date__gte=last_14_days, date__lt=last_7_days)
    
    # Improvement calculations (combine both sources)
    improvement_data = {}
    recent_7_count = recent_7_feedback.count() + recent_7_sessions.count()
    previous_7_count = previous_7_feedback.count() + previous_7_sessions.count()
    
    if recent_7_count > 0 and previous_7_count > 0:
        # Calculate averages from both sources
        recent_fb_avg_filler = recent_7_feedback.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
        recent_fb_avg_clarity = recent_7_feedback.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
        recent_fb_avg_pacing = recent_7_feedback.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
        
        recent_sess_avg_filler = recent_7_sessions.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
        recent_sess_avg_clarity = recent_7_sessions.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
        recent_sess_avg_pacing = recent_7_sessions.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
        
        prev_fb_avg_filler = previous_7_feedback.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
        prev_fb_avg_clarity = previous_7_feedback.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
        prev_fb_avg_pacing = previous_7_feedback.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
        
        prev_sess_avg_filler = previous_7_sessions.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
        prev_sess_avg_clarity = previous_7_sessions.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
        prev_sess_avg_pacing = previous_7_sessions.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
        
        # Weighted averages
        recent_total = recent_7_feedback.count() + recent_7_sessions.count()
        prev_total = previous_7_feedback.count() + previous_7_sessions.count()
        
        if recent_total > 0:
            recent_avg_filler = ((recent_fb_avg_filler * recent_7_feedback.count()) + (recent_sess_avg_filler * recent_7_sessions.count())) / recent_total
            recent_avg_clarity = ((recent_fb_avg_clarity * recent_7_feedback.count()) + (recent_sess_avg_clarity * recent_7_sessions.count())) / recent_total
            recent_avg_pacing = ((recent_fb_avg_pacing * recent_7_feedback.count()) + (recent_sess_avg_pacing * recent_7_sessions.count())) / recent_total
        else:
            recent_avg_filler = recent_fb_avg_filler or recent_sess_avg_filler or 0
            recent_avg_clarity = recent_fb_avg_clarity or recent_sess_avg_clarity or 0
            recent_avg_pacing = recent_fb_avg_pacing or recent_sess_avg_pacing or 0
        
        if prev_total > 0:
            previous_avg_filler = ((prev_fb_avg_filler * previous_7_feedback.count()) + (prev_sess_avg_filler * previous_7_sessions.count())) / prev_total
            previous_avg_clarity = ((prev_fb_avg_clarity * previous_7_feedback.count()) + (prev_sess_avg_clarity * previous_7_sessions.count())) / prev_total
            previous_avg_pacing = ((prev_fb_avg_pacing * previous_7_feedback.count()) + (prev_sess_avg_pacing * previous_7_sessions.count())) / prev_total
        else:
            previous_avg_filler = prev_fb_avg_filler or prev_sess_avg_filler or 0
            previous_avg_clarity = prev_fb_avg_clarity or prev_sess_avg_clarity or 0
            previous_avg_pacing = prev_fb_avg_pacing or prev_sess_avg_pacing or 0
        
        # Calculate percentage improvement (lower is better for all scores)
        improvement_data = {
            'filler_improvement': ((previous_avg_filler - recent_avg_filler) / previous_avg_filler * 100) if previous_avg_filler > 0 else 0,
            'clarity_improvement': ((previous_avg_clarity - recent_avg_clarity) / previous_avg_clarity * 100) if previous_avg_clarity > 0 else 0,
            'pacing_improvement': ((previous_avg_pacing - recent_avg_pacing) / previous_avg_pacing * 100) if previous_avg_pacing > 0 else 0,
        }
    
    # Get chart data (last 30 days, daily aggregates) - combine Feedback and SpeechSession
    chart_data = []
    for i in range(29, -1, -1):
        date = (now - timedelta(days=i)).date()
        day_feedbacks = feedbacks.filter(
            created_at__date=date,
            filler_score__isnull=False
        )
        day_sessions = sessions_with_scores.filter(date__date=date)
        
        # Combine data from both sources for this day
        day_fb_count = day_feedbacks.count()
        day_sess_count = day_sessions.count()
        
        if day_fb_count > 0 or day_sess_count > 0:
            # Get averages from each source
            fb_avg_filler = day_feedbacks.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            fb_avg_clarity = day_feedbacks.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            fb_avg_pacing = day_feedbacks.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            sess_avg_filler = day_sessions.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            sess_avg_clarity = day_sessions.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            sess_avg_pacing = day_sessions.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            # Weighted average
            total_count = day_fb_count + day_sess_count
            if total_count > 0:
                combined_filler = ((fb_avg_filler * day_fb_count) + (sess_avg_filler * day_sess_count)) / total_count
                combined_clarity = ((fb_avg_clarity * day_fb_count) + (sess_avg_clarity * day_sess_count)) / total_count
                combined_pacing = ((fb_avg_pacing * day_fb_count) + (sess_avg_pacing * day_sess_count)) / total_count
            else:
                combined_filler = fb_avg_filler or sess_avg_filler or 0
                combined_clarity = fb_avg_clarity or sess_avg_clarity or 0
                combined_pacing = fb_avg_pacing or sess_avg_pacing or 0
            
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'filler_score': float(combined_filler),
                'clarity_score': float(combined_clarity),
                'pacing_score': float(combined_pacing),
            })
    
    # Convert chart_data to JSON for template
    import json
    chart_data_json = json.dumps(chart_data)
    
    # Identify worst area for drill recommendations
    worst_area = 'filler_words'  # Default
    if avg_filler_score < 0.4 and avg_clarity_score > 0.6:
        worst_area = 'pronunciation'
    elif avg_clarity_score < 0.4 and avg_pacing_score > 0.6:
        worst_area = 'pacing'
    elif avg_pacing_score > 0.6:
        worst_area = 'pacing'
    
    # Recommend drills based on worst area
    from .utils import DrillRecommender
    recommender = DrillRecommender()
    history = list(feedbacks.order_by('-created_at')[:7])
    recommended_drills = Drill.objects.filter(
        is_active=True,
        skill_type=worst_area
    ).order_by('?')[:3]  # Random 3 from worst area
    
    # If no recommendations, get general drills
    if not recommended_drills.exists():
        recommended_drills = Drill.objects.filter(is_active=True).order_by('?')[:3]
    
    # Get recent activity (combine feedbacks, sessions, drills)
    recent_activity = []
    
    # Add recent feedback
    for fb in feedbacks[:10]:
        recent_activity.append({
            'type': 'feedback',
            'date': fb.created_at,
            'title': 'Speech Analysis',
            'data': fb,
            'icon': 'fa-microphone',
            'color': 'blue',
        })
    
    # Add recent sessions
    for session in sessions[:10]:
        recent_activity.append({
            'type': 'session',
            'date': session.date,
            'title': f'Session ({session.duration}s)',
            'data': session,
            'icon': 'fa-headphones',
            'color': 'green',
        })
    
    # Add recent drill completions
    for completion in drill_completions[:10]:
        recent_activity.append({
            'type': 'drill',
            'date': completion.completed_at,
            'title': completion.drill.name,
            'data': completion,
            'icon': 'fa-dumbbell',
            'color': 'purple',
        })
    
    # Sort by date and get most recent 10
    recent_activity.sort(key=lambda x: x['date'], reverse=True)
    recent_activity = recent_activity[:10]
    
    # Generate encouraging message
    encouraging_message = None
    if improvement_data and improvement_data.get('filler_improvement', 0) > 20:
        encouraging_message = {
            'type': 'success',
            'title': 'Great Progress!',
            'message': f"You've reduced fillers by {improvement_data['filler_improvement']:.0f}% this week! Keep it up!",
        }
    elif improvement_data and improvement_data.get('clarity_improvement', 0) > 15:
        encouraging_message = {
            'type': 'success',
            'title': 'Clarity Improving!',
            'message': f"Your speech clarity improved by {improvement_data['clarity_improvement']:.0f}% this week!",
        }
    elif feedbacks.count() == 0 and sessions.count() == 0:
        encouraging_message = {
            'type': 'info',
            'title': 'Get Started!',
            'message': 'Record your first speech session to start tracking your progress and receive personalized feedback!',
        }
    elif (feedbacks.count() + sessions.count()) < 5:
        encouraging_message = {
            'type': 'info',
            'title': 'Keep Going!',
            'message': 'You\'re just getting started! Record more sessions to see your improvement trends.',
        }
    else:
        # Calculate potential based on current scores
        if avg_filler_score > 0.6:
            encouraging_message = {
                'type': 'warning',
                'title': 'Focus Area',
                'message': f'Your filler score is {avg_filler_score*100:.0f}%. With practice, you can reduce fillers by 50%+ in the next 2 weeks!',
            }
        elif avg_clarity_score > 0.5:
            encouraging_message = {
                'type': 'warning',
                'title': 'Focus Area',
                'message': f'Your clarity score is {avg_clarity_score*100:.0f}%. Practice pronunciation drills to improve clarity!',
            }
        else:
            encouraging_message = {
                'type': 'success',
                'title': 'Great Job!',
                'message': 'You\'re making steady progress! Keep practicing to maintain and improve your scores.',
            }
    
    # Get all drills for listing
    all_drills = Drill.objects.filter(is_active=True).order_by('skill_type', 'name')
    
    context = {
        # Latest Scores (most recent session/feedback) - Option 2
        'latest_filler_score': round(latest_filler_score * 100, 1) if latest_filler_score is not None else None,
        'latest_clarity_score': round(latest_clarity_score * 100, 1) if latest_clarity_score is not None else None,
        'latest_pacing_score': round(latest_pacing_score * 100, 1) if latest_pacing_score is not None else None,
        
        # Average Scores (all-time from both sources) - Option 2
        'avg_filler_score': round(avg_filler_score * 100, 1) if avg_filler_score else None,
        'avg_clarity_score': round(avg_clarity_score * 100, 1) if avg_clarity_score else None,
        'avg_pacing_score': round(avg_pacing_score * 100, 1) if avg_pacing_score else None,
        'avg_f1_score': round(avg_f1_score, 2),
        'avg_fillers': round(avg_fillers, 1),
        'avg_wpm': round(avg_wpm, 1),
        
        # Counts
        'total_sessions': feedbacks.count() + sessions.count(),
        'total_drill_completions': drill_completions.count(),
        'total_feedbacks': feedbacks.count(),
        
        # Charts & Analytics
        'chart_data': chart_data,
        'chart_data_json': chart_data_json,
        'improvement_data': improvement_data,
        'worst_area': worst_area,
        
        # Recommendations
        'recommended_drills': recommended_drills,
        'encouraging_message': encouraging_message,
        
        # Activity
        'recent_activity': recent_activity,
        
        # Lists
        'recent_feedback': feedbacks[:5],
        'drills': all_drills,
    }
    
    return render(request, 'coach/dashboard.html', context)


@login_required
def upload_view(request):
    """Handle audio file upload for analysis."""
    if request.method == 'POST':
        # This will be implemented in future sprints
        # For now, just show a placeholder message
        messages.info(request, 'Audio upload functionality will be implemented in Sprint 2.')
        return redirect('coach:dashboard')
    
    return render(request, 'coach/upload.html')


@login_required
def record_view(request):
    """Display voice recording interface with real-time analysis."""
    return render(request, 'coach/record.html')


@login_required
def reports_view(request):
    """Display comprehensive reports with charts and analytics."""
    from speech_sessions.models import SpeechSession
    from datetime import timedelta
    import json
    
    # Get all feedback from Feedback model
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    
    # Get all analyzed speech sessions with scores
    speech_sessions = SpeechSession.objects.filter(
        user=request.user,
        status='analyzed'
    ).order_by('-date')
    
    sessions_with_scores = speech_sessions.filter(filler_score__isnull=False)
    
    # Combine both data sources into a unified list
    all_feedback_data = []
    
    # Add Feedback entries
    for fb in feedbacks:
        all_feedback_data.append({
            'type': 'feedback',
            'id': fb.id,
            'date': fb.created_at,
            'filler_score': fb.filler_score,
            'clarity_score': fb.clarity_score,
            'pacing_score': fb.pacing_score,
        })
    
    # Add SpeechSession entries
    for session in speech_sessions:
        all_feedback_data.append({
            'type': 'session',
            'id': session.id,
            'date': session.date,
            'filler_score': session.filler_score,
            'clarity_score': session.clarity_score,
            'pacing_score': session.pacing_score,
        })
    
    # Calculate average scores
    scores_with_values = [f for f in all_feedback_data if f['filler_score'] is not None]
    if scores_with_values:
        avg_filler = sum(f['filler_score'] for f in scores_with_values) / len(scores_with_values)
        avg_clarity = sum(f['clarity_score'] for f in scores_with_values if f['clarity_score']) / len([f for f in scores_with_values if f['clarity_score']]) if any(f['clarity_score'] for f in scores_with_values) else None
        avg_pacing = sum(f['pacing_score'] for f in scores_with_values if f['pacing_score']) / len([f for f in scores_with_values if f['pacing_score']]) if any(f['pacing_score'] for f in scores_with_values) else None
    else:
        avg_filler = None
        avg_clarity = None
        avg_pacing = None
    
    # Get chart data for progress (last 30 days) - same logic as dashboard
    now = timezone.now()
    chart_data = []
    
    for i in range(29, -1, -1):
        date = (now - timedelta(days=i)).date()
        day_feedbacks = feedbacks.filter(
            created_at__date=date,
            filler_score__isnull=False
        )
        day_sessions = sessions_with_scores.filter(date__date=date)
        
        # Combine data from both sources for this day
        day_fb_count = day_feedbacks.count()
        day_sess_count = day_sessions.count()
        
        if day_fb_count > 0 or day_sess_count > 0:
            # Get averages from each source
            fb_avg_filler = day_feedbacks.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            fb_avg_clarity = day_feedbacks.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            fb_avg_pacing = day_feedbacks.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            sess_avg_filler = day_sessions.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            sess_avg_clarity = day_sessions.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            sess_avg_pacing = day_sessions.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            # Weighted average
            total_count = day_fb_count + day_sess_count
            if total_count > 0:
                combined_filler = ((fb_avg_filler * day_fb_count) + (sess_avg_filler * day_sess_count)) / total_count
                combined_clarity = ((fb_avg_clarity * day_fb_count) + (sess_avg_clarity * day_sess_count)) / total_count
                combined_pacing = ((fb_avg_pacing * day_fb_count) + (sess_avg_pacing * day_sess_count)) / total_count
            else:
                combined_filler = fb_avg_filler or sess_avg_filler or 0
                combined_clarity = fb_avg_clarity or sess_avg_clarity or 0
                combined_pacing = fb_avg_pacing or sess_avg_pacing or 0
            
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'filler_score': float(combined_filler),
                'clarity_score': float(combined_clarity),
                'pacing_score': float(combined_pacing),
            })
    
    chart_data_json = json.dumps(chart_data) if chart_data else None
    
    # Additional chart data for distribution and trends
    distribution_data = {
        'filler_scores': [f['filler_score'] * 100 for f in scores_with_values if f['filler_score'] is not None],
        'clarity_scores': [f['clarity_score'] * 100 for f in scores_with_values if f['clarity_score'] is not None],
        'pacing_scores': [f['pacing_score'] * 100 for f in scores_with_values if f['pacing_score'] is not None],
    }
    distribution_data_json = json.dumps(distribution_data)
    
    # Weekly improvement trends (last 4 weeks)
    weekly_trends = []
    for week in range(4):
        week_start = now - timedelta(days=(week + 1) * 7)
        week_end = now - timedelta(days=week * 7)
        week_feedbacks = feedbacks.filter(
            created_at__gte=week_start,
            created_at__lt=week_end,
            filler_score__isnull=False
        )
        week_sessions = sessions_with_scores.filter(
            date__gte=week_start,
            date__lt=week_end
        )
        
        week_fb_count = week_feedbacks.count()
        week_sess_count = week_sessions.count()
        
        if week_fb_count > 0 or week_sess_count > 0:
            fb_avg_f = week_feedbacks.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            fb_avg_c = week_feedbacks.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            fb_avg_p = week_feedbacks.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            sess_avg_f = week_sessions.aggregate(avg=models.Avg('filler_score'))['avg'] or 0
            sess_avg_c = week_sessions.aggregate(avg=models.Avg('clarity_score'))['avg'] or 0
            sess_avg_p = week_sessions.aggregate(avg=models.Avg('pacing_score'))['avg'] or 0
            
            total_w = week_fb_count + week_sess_count
            if total_w > 0:
                weekly_trends.append({
                    'week': f'Week {4 - week}',
                    'filler_score': float(((fb_avg_f * week_fb_count) + (sess_avg_f * week_sess_count)) / total_w),
                    'clarity_score': float(((fb_avg_c * week_fb_count) + (sess_avg_c * week_sess_count)) / total_w),
                    'pacing_score': float(((fb_avg_p * week_fb_count) + (sess_avg_p * week_sess_count)) / total_w),
                })
    weekly_trends_json = json.dumps(weekly_trends) if weekly_trends else None
    
    # Calculate statistics
    stats = {
        'total_sessions': len(all_feedback_data),
        'total_feedbacks': feedbacks.count(),
        'total_speech_sessions': speech_sessions.count(),
    }
    
    context = {
        'stats': stats,
        'avg_filler_score': round(avg_filler * 100, 1) if avg_filler else None,
        'avg_clarity_score': round(avg_clarity * 100, 1) if avg_clarity else None,
        'avg_pacing_score': round(avg_pacing * 100, 1) if avg_pacing else None,
        'chart_data': chart_data,
        'chart_data_json': chart_data_json,
        'distribution_data_json': distribution_data_json,
        'weekly_trends_json': weekly_trends_json,
        'has_data': len(all_feedback_data) > 0,
    }
    
    return render(request, 'coach/reports.html', context)


@login_required
def feedback_view(request, feedback_id=None):
    """Display detailed feedback analysis from both Feedback and SpeechSession models."""
    from speech_sessions.models import SpeechSession
    from datetime import timedelta
    import json
    
    if feedback_id:
        try:
            feedback = Feedback.objects.get(id=feedback_id, user=request.user)
            context = {'feedback': feedback}
            return render(request, 'coach/feedback.html', context)
        except Feedback.DoesNotExist:
            messages.error(request, 'Feedback not found.')
            return redirect('coach:dashboard')
    
    # Get all feedback from Feedback model
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    
    # Get all analyzed speech sessions
    speech_sessions = SpeechSession.objects.filter(
        user=request.user,
        status='analyzed'
    ).order_by('-date')
    
    # Combine both data sources into a unified list
    all_feedback_data = []
    
    # Add Feedback entries
    for fb in feedbacks:
        all_feedback_data.append({
            'type': 'feedback',
            'id': fb.id,
            'date': fb.created_at,
            'title': f'Speech Analysis #{fb.id}',
            'data': fb,
            'filler_score': fb.filler_score,
            'clarity_score': fb.clarity_score,
            'pacing_score': fb.pacing_score,
            'fillers_count': fb.fillers_count,
            'wpm': fb.wpm,
            'transcription': fb.transcription,
            'audio_file': fb.audio_file,
        })
    
    # Add SpeechSession entries (now with scores directly from model)
    for session in speech_sessions:
        all_feedback_data.append({
            'type': 'session',
            'id': session.id,
            'date': session.date,
            'title': f'Session #{session.id}',
            'data': session,
            'filler_score': session.filler_score,
            'clarity_score': session.clarity_score,
            'pacing_score': session.pacing_score,
            'fillers_count': session.filler_count,
            'wpm': None,  # WPM not directly stored, would need to calculate from transcript
            'transcription': session.transcription,
            'audio_file': session.audio_file,
            'duration': session.duration,
        })
    
    # Sort by date (most recent first)
    all_feedback_data.sort(key=lambda x: x['date'], reverse=True)
    
    # Calculate statistics from combined data
    stats = {
        'total_sessions': len(all_feedback_data),
        'total_feedbacks': feedbacks.count(),
        'total_speech_sessions': speech_sessions.count(),
    }
    
    # Calculate average scores from entries that have them (same logic as dashboard)
    scores_with_values = [f for f in all_feedback_data if f['filler_score'] is not None]
    if scores_with_values:
        avg_filler = sum(f['filler_score'] for f in scores_with_values) / len(scores_with_values)
        avg_clarity = sum(f['clarity_score'] for f in scores_with_values if f['clarity_score']) / len([f for f in scores_with_values if f['clarity_score']]) if any(f['clarity_score'] for f in scores_with_values) else None
        avg_pacing = sum(f['pacing_score'] for f in scores_with_values if f['pacing_score']) / len([f for f in scores_with_values if f['pacing_score']]) if any(f['pacing_score'] for f in scores_with_values) else None
    else:
        avg_filler = None
        avg_clarity = None
        avg_pacing = None
    
    context = {
        'feedbacks': all_feedback_data,  # Combined list
        'stats': stats,
        'avg_filler_score': round(avg_filler * 100, 1) if avg_filler else None,
        'avg_clarity_score': round(avg_clarity * 100, 1) if avg_clarity else None,
        'avg_pacing_score': round(avg_pacing * 100, 1) if avg_pacing else None,
    }
    
    return render(request, 'coach/feedback.html', context)


@login_required
def export_feedback_report(request, format='pdf'):
    """Export comprehensive feedback report as PDF or CSV."""
    from speech_sessions.models import SpeechSession
    from datetime import timedelta
    from django.http import HttpResponse
    import csv
    import io
    
    # Get all data (same as feedback_view)
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    speech_sessions = SpeechSession.objects.filter(
        user=request.user,
        status='analyzed'
    ).order_by('-date')
    sessions_with_scores = speech_sessions.filter(filler_score__isnull=False)
    
    # Combine data
    all_feedback_data = []
    for fb in feedbacks:
        all_feedback_data.append({
            'type': 'Feedback',
            'id': fb.id,
            'date': fb.created_at,
            'filler_score': fb.filler_score * 100 if fb.filler_score else None,
            'clarity_score': fb.clarity_score * 100 if fb.clarity_score else None,
            'pacing_score': fb.pacing_score * 100 if fb.pacing_score else None,
            'fillers_count': fb.fillers_count,
            'wpm': fb.wpm,
        })
    
    for session in speech_sessions:
        all_feedback_data.append({
            'type': 'Session',
            'id': session.id,
            'date': session.date,
            'filler_score': session.filler_score * 100 if session.filler_score else None,
            'clarity_score': session.clarity_score * 100 if session.clarity_score else None,
            'pacing_score': session.pacing_score * 100 if session.pacing_score else None,
            'fillers_count': session.filler_count,
            'wpm': None,
        })
    
    all_feedback_data.sort(key=lambda x: x['date'], reverse=True)
    
    # Calculate averages
    scores_with_values = [f for f in all_feedback_data if f['filler_score'] is not None]
    if scores_with_values:
        avg_filler = sum(f['filler_score'] for f in scores_with_values) / len(scores_with_values)
        avg_clarity = sum(f['clarity_score'] for f in scores_with_values if f['clarity_score']) / len([f for f in scores_with_values if f['clarity_score']]) if any(f['clarity_score'] for f in scores_with_values) else None
        avg_pacing = sum(f['pacing_score'] for f in scores_with_values if f['pacing_score']) / len([f for f in scores_with_values if f['pacing_score']]) if any(f['pacing_score'] for f in scores_with_values) else None
    else:
        avg_filler = avg_clarity = avg_pacing = None
    
    if format == 'csv':
        # Generate CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="speech_analysis_report_{request.user.email}_{timezone.now().strftime("%Y%m%d")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Speech Analysis Report'])
        writer.writerow(['Generated for:', request.user.email])
        writer.writerow(['Generated on:', timezone.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        writer.writerow(['Summary Statistics'])
        writer.writerow(['Total Entries', len(all_feedback_data)])
        writer.writerow(['Average Filler Score (%)', f'{avg_filler:.2f}' if avg_filler else 'N/A'])
        writer.writerow(['Average Clarity Score (%)', f'{avg_clarity:.2f}' if avg_clarity else 'N/A'])
        writer.writerow(['Average Pacing Score (%)', f'{avg_pacing:.2f}' if avg_pacing else 'N/A'])
        writer.writerow([])
        writer.writerow(['Detailed Data'])
        writer.writerow(['Type', 'ID', 'Date', 'Filler Score (%)', 'Clarity Score (%)', 'Pacing Score (%)', 'Fillers Count', 'WPM'])
        
        for entry in all_feedback_data:
            writer.writerow([
                entry['type'],
                entry['id'],
                entry['date'].strftime('%Y-%m-%d %H:%M:%S'),
                f"{entry['filler_score']:.2f}" if entry['filler_score'] else 'N/A',
                f"{entry['clarity_score']:.2f}" if entry['clarity_score'] else 'N/A',
                f"{entry['pacing_score']:.2f}" if entry['pacing_score'] else 'N/A',
                entry['fillers_count'] or 'N/A',
                entry['wpm'] or 'N/A',
            ])
        
        return response
    
    elif format == 'pdf':
        # Generate PDF using reportlab
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph('Speech Analysis Report', title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Header info
            header_style = ParagraphStyle(
                'Header',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.grey,
                alignment=TA_LEFT
            )
            story.append(Paragraph(f'<b>User:</b> {request.user.get_full_name()} ({request.user.email})', header_style))
            story.append(Paragraph(f'<b>Generated:</b> {timezone.now().strftime("%B %d, %Y at %I:%M %p")}', header_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Summary Statistics
            summary_style = ParagraphStyle(
                'Summary',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=12
            )
            story.append(Paragraph('Summary Statistics', summary_style))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Entries', str(len(all_feedback_data))],
                ['Average Filler Score (%)', f'{avg_filler:.2f}' if avg_filler else 'N/A'],
                ['Average Clarity Score (%)', f'{avg_clarity:.2f}' if avg_clarity else 'N/A'],
                ['Average Pacing Score (%)', f'{avg_pacing:.2f}' if avg_pacing else 'N/A'],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Data
            story.append(Paragraph('Detailed Session Data', summary_style))
            
            # Prepare data table (limit to 50 entries for PDF)
            table_data = [['Type', 'ID', 'Date', 'Filler (%)', 'Clarity (%)', 'Pacing (%)', 'Fillers', 'WPM']]
            for entry in all_feedback_data[:50]:  # Limit to 50 for PDF
                table_data.append([
                    entry['type'],
                    str(entry['id']),
                    entry['date'].strftime('%Y-%m-%d'),
                    f"{entry['filler_score']:.1f}" if entry['filler_score'] else 'N/A',
                    f"{entry['clarity_score']:.1f}" if entry['clarity_score'] else 'N/A',
                    f"{entry['pacing_score']:.1f}" if entry['pacing_score'] else 'N/A',
                    str(entry['fillers_count']) if entry['fillers_count'] else 'N/A',
                    str(entry['wpm']) if entry['wpm'] else 'N/A',
                ])
            
            if len(all_feedback_data) > 50:
                table_data.append(['', '', f'... and {len(all_feedback_data) - 50} more entries', '', '', '', '', ''])
            
            data_table = Table(table_data, colWidths=[0.8*inch, 0.6*inch, 1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.6*inch])
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(data_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            response = HttpResponse(buffer.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="speech_analysis_report_{request.user.email}_{timezone.now().strftime("%Y%m%d")}.pdf"'
            return response
            
        except ImportError:
            messages.error(request, 'PDF generation requires reportlab. Please install it: pip install reportlab')
            return redirect('coach:feedback')
    
    else:
        messages.error(request, 'Invalid export format.')
        return redirect('coach:feedback')


@login_required
def profile_view(request):
    """Display and edit user profile."""
    if request.method == 'POST':
        # Handle profile updates
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        
        if first_name and last_name:
            request.user.first_name = first_name
            request.user.last_name = last_name
            request.user.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('coach:profile')
        else:
            messages.error(request, 'Please fill in all required fields.')
    
    return render(request, 'coach/profile.html')


# API endpoints for future AJAX functionality
@login_required
@require_http_methods(["GET"])
def api_user_stats(request):
    """API endpoint to get user statistics."""
    feedbacks = Feedback.objects.filter(user=request.user)
    
    stats = {
        'total_sessions': feedbacks.count(),
        'avg_f1_score': feedbacks.aggregate(avg_f1=models.Avg('f1_score'))['avg_f1'] or 0,
        'avg_fillers': feedbacks.aggregate(avg_fillers=models.Avg('fillers_count'))['avg_fillers'] or 0,
        'avg_wpm': feedbacks.aggregate(avg_wpm=models.Avg('wpm'))['avg_wpm'] or 0,
        'best_f1_score': feedbacks.aggregate(best_f1=models.Max('f1_score'))['best_f1'] or 0,
        'recent_trend': list(feedbacks.order_by('-created_at')[:10].values('f1_score', 'created_at'))
    }
    
    return JsonResponse(stats)


# 2FA Views
@login_required
def security_settings_view(request):
    """Display and manage security settings including email verification and 2FA."""
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'send_verification':
            if request.user.is_email_verified:
                messages.info(request, 'Your email is already verified.')
            else:
                if send_verification_email(request.user):
                    messages.success(request, 'Verification email sent! Please check your inbox.')
                else:
                    messages.error(request, 'Failed to send verification email. Please try again.')
        
        elif action == 'enable_2fa':
            if not request.user.is_email_verified:
                messages.error(request, 'Please verify your email address before enabling 2FA.')
            elif not request.user.has_2fa_setup():
                return redirect('coach:setup_2fa')
            else:
                messages.error(request, '2FA is already enabled.')
        
        elif action == 'disable_2fa':
            # Disable 2FA
            request.user.is_2fa_enabled = False
            request.user.save()
            
            # Remove TOTP device
            device = request.user.get_totp_device()
            if device:
                device.delete()
            
            messages.success(request, '2FA has been disabled successfully.')
        
        return redirect('coach:security_settings')
    
    context = {
        'has_2fa_setup': request.user.has_2fa_setup(),
        'is_2fa_enabled': request.user.is_2fa_enabled,
        'is_email_verified': request.user.is_email_verified,
    }
    
    return render(request, 'coach/security_settings.html', context)


@login_required
def setup_2fa_view(request):
    """Setup 2FA with QR code generation."""
    if request.user.has_2fa_setup():
        messages.info(request, '2FA is already set up.')
        return redirect('coach:security_settings')
    
    if request.method == 'POST':
        # Create TOTP device
        device = TOTPDevice.objects.create(
            user=request.user,
            name=f"{request.user.email} - TOTP Device",
            confirmed=False
        )
        
        # Generate QR code
        qr_code_data = device.config_url
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_code_data)
        qr.make(fit=True)
        
        # Create SVG QR code
        factory = qrcode.image.svg.SvgPathImage
        img = qr.make_image(image_factory=factory)
        
        # Convert to base64 for embedding in HTML
        buffer = io.BytesIO()
        img.save(buffer)
        qr_code_svg = buffer.getvalue().decode()
        
        context = {
            'device': device,
            'qr_code_svg': qr_code_svg,
            'manual_key': device.bin_key.hex().upper(),
        }
        
        return render(request, 'coach/2fa_setup.html', context)
    
    return render(request, 'coach/2fa_setup.html')


@login_required
def verify_2fa_setup_view(request):
    """Verify 2FA setup with TOTP code."""
    if request.method == 'POST':
        totp_code = request.POST.get('totp_code', '').strip()
        
        if not totp_code:
            messages.error(request, 'Please enter a TOTP code.')
            return redirect('coach:setup_2fa')
        
        # Get the unconfirmed device
        device = TOTPDevice.objects.filter(user=request.user, confirmed=False).first()
        
        if not device:
            messages.error(request, 'No pending 2FA setup found.')
            return redirect('coach:security_settings')
        
        # Verify the code
        if device.verify_token(totp_code):
            device.confirmed = True
            device.save()
            
            # Enable 2FA for user
            request.user.is_2fa_enabled = True
            request.user.save()
            
            # Generate backup codes
            backup_codes = request.user.generate_backup_codes()
            
            messages.success(request, '2FA has been successfully enabled!')
            return render(request, 'coach/2fa_backup_codes.html', {
                'backup_codes': backup_codes
            })
        else:
            messages.error(request, 'Invalid TOTP code. Please try again.')
    
    return redirect('coach:setup_2fa')


def verify_2fa_view(request):
    """Verify 2FA during login."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if not user.is_2fa_enabled or not user.has_2fa_setup():
        messages.error(request, '2FA is not properly set up.')
        return redirect('coach:login')
    
    if request.method == 'POST':
        totp_code = request.POST.get('totp_code', '').strip()
        backup_code = request.POST.get('backup_code', '').strip()
        
        if totp_code:
            # Verify TOTP code
            device = user.get_totp_device()
            if device and device.verify_token(totp_code):
                # Complete login
                login(request, user)
                request.session.pop('temp_user_id', None)
                messages.success(request, '2FA verification successful!')
                return redirect('coach:dashboard')
            else:
                messages.error(request, 'Invalid TOTP code. Please try again.')
        
        elif backup_code:
            # Verify backup code
            if backup_code in user.backup_codes:
                # Remove used backup code
                user.backup_codes.remove(backup_code)
                user.save()
                
                # Complete login
                login(request, user)
                request.session.pop('temp_user_id', None)
                messages.success(request, 'Backup code accepted!')
                return redirect('coach:dashboard')
            else:
                messages.error(request, 'Invalid backup code. Please try again.')
        
        else:
            messages.error(request, 'Please enter either a TOTP code or backup code.')
    
    context = {
        'user': user
    }
    return render(request, 'coach/2fa_verify.html', context)


def send_fallback_code_view(request):
    """Send fallback code via email."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if request.method == 'POST':
        # Generate a simple fallback code (in production, this would be more secure)
        import random
        fallback_code = str(random.randint(100000, 999999))
        
        # Store in session for verification
        request.session['fallback_code'] = fallback_code
        request.session['fallback_code_expires'] = 300  # 5 minutes
        
        # Send email
        try:
            send_mail(
                'Verbal Coach - 2FA Fallback Code',
                f'Your 2FA fallback code is: {fallback_code}\n\nThis code will expire in 5 minutes.',
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                fail_silently=False,
            )
            messages.success(request, 'Fallback code sent to your email address.')
        except Exception as e:
            messages.error(request, f'Failed to send fallback code: {str(e)}')
    
    return redirect('coach:verify_2fa')


def verify_fallback_code_view(request):
    """Verify fallback code."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if request.method == 'POST':
        fallback_code = request.POST.get('fallback_code', '').strip()
        stored_code = request.session.get('fallback_code')
        expires = request.session.get('fallback_code_expires', 0)
        
        if not stored_code or expires <= 0:
            messages.error(request, 'No valid fallback code found. Please request a new one.')
            return redirect('coach:verify_2fa')
        
        if fallback_code == stored_code:
            # Clear the fallback code
            del request.session['fallback_code']
            del request.session['fallback_code_expires']
            
            # Complete login
            login(request, user)
            request.session.pop('temp_user_id', None)
            messages.success(request, 'Fallback code verified successfully!')
            return redirect('coach:dashboard')
        else:
            messages.error(request, 'Invalid fallback code. Please try again.')
    
    return redirect('coach:verify_2fa')


# Email Verification Views
def verification_sent_view(request):
    """Show verification email sent message."""
    return render(request, 'coach/verification_sent.html')


def verify_email_view(request, token):
    """Verify email address using token."""
    user = verify_email_token(token)
    
    if user:
        user.is_email_verified = True
        user.is_active = True
        user.save()
        messages.success(request, 'Email verified successfully! You can now log in to your account.')
        return redirect('coach:email_verified')
    else:
        messages.error(request, 'Invalid or expired verification link. Please request a new one.')
        return redirect('coach:resend_verification')


def email_verified_view(request):
    """Show email verified confirmation."""
    return render(request, 'coach/email_verified.html')


def resend_verification_view(request):
    """Resend email verification."""
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            if user.is_email_verified:
                messages.info(request, 'Your email is already verified.')
                return redirect('coach:login')
            
            if send_verification_email(user):
                messages.success(request, 'Verification email sent! Please check your inbox.')
            else:
                messages.error(request, 'Failed to send verification email. Please try again.')
        except User.DoesNotExist:
            messages.error(request, 'No account found with this email address.')
    
    return render(request, 'coach/resend_verification.html')


# 2FA Choice Views
def choose_2fa_method_view(request):
    """Allow user to choose 2FA method (TOTP or Email OTP)."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if not user.is_2fa_enabled or not user.has_2fa_setup():
        messages.error(request, '2FA is not properly set up.')
        return redirect('coach:login')
    
    if request.method == 'POST':
        form = TwoFactorChoiceForm(request.POST)
        if form.is_valid():
            method = form.cleaned_data['method']
            if method == 'totp':
                return redirect('coach:verify_2fa')
            elif method == 'email':
                # Generate and send OTP
                otp = create_email_otp(user)
                if send_otp_email(user, otp.code):
                    messages.success(request, 'OTP code sent to your email address.')
                    return redirect('coach:verify_otp')
                else:
                    messages.error(request, 'Failed to send OTP. Please try again.')
            elif method == 'passkey':
                # Check if user has passkey
                if user.has_passkey():
                    return redirect('coach:verify_passkey')
                else:
                    messages.error(request, 'You have not registered any passkeys yet.')
    else:
        form = TwoFactorChoiceForm()
    
    context = {
        'form': form,
        'user': user
    }
    return render(request, 'coach/choose_2fa_method.html', context)


def verify_otp_view(request):
    """Verify email OTP code."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if request.method == 'POST':
        form = OTPVerificationForm(request.POST)
        if form.is_valid():
            otp_code = form.cleaned_data['otp_code']
            
            # Find valid OTP for user
            otp = EmailOTP.objects.filter(
                user=user,
                code=otp_code,
                is_used=False
            ).first()
            
            if otp and otp.is_valid():
                otp.is_used = True
                otp.save()
                
                # Complete login
                login(request, user)
                request.session.pop('temp_user_id', None)
                messages.success(request, 'Login successful!')
                return redirect('coach:dashboard')
            else:
                messages.error(request, 'Invalid or expired OTP code.')
    else:
        form = OTPVerificationForm()
    
    context = {
        'form': form,
        'user': user
    }
    return render(request, 'coach/verify_otp.html', context)


# Password Reset Views
class CustomPasswordResetView(PasswordResetView):
    """Custom password reset view."""
    form_class = CustomPasswordResetForm
    template_name = 'coach/password_reset.html'
    email_template_name = 'emails/password_reset.html'
    subject_template_name = 'emails/password_reset_subject.txt'
    success_url = reverse_lazy('coach:password_reset_done')


class CustomPasswordResetDoneView(PasswordResetDoneView):
    """Custom password reset done view."""
    template_name = 'coach/password_reset_done.html'


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    """Custom password reset confirm view."""
    template_name = 'coach/password_reset_confirm.html'
    success_url = reverse_lazy('coach:password_reset_complete')


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    """Custom password reset complete view."""
    template_name = 'coach/password_reset_complete.html'


# Passkey (WebAuthn) Views
def get_rp_id(request):
    """Get the Relying Party ID from the request host.
    
    WebAuthn requires the RP ID to exactly match the origin's hostname (without port).
    For local development:
    - If accessing via localhost:8000, use 'localhost'
    - If accessing via 127.0.0.1:8000, use '127.0.0.1'
    
    The key is consistency - use whatever the browser's origin hostname is.
    """
    host = request.get_host().split(':')[0]  # Remove port
    # Use the hostname as-is - don't convert between localhost and 127.0.0.1
    # The browser treats them as different origins
    return host


def get_origin(request):
    """Get the origin from the request.
    
    The origin must exactly match what the browser sees (including protocol and port).
    """
    scheme = 'https' if request.is_secure() else 'http'
    host = request.get_host()  # Includes port (e.g., '127.0.0.1:8000' or 'localhost:8000')
    return f"{scheme}://{host}"


def safe_b64decode(data):
    """Safely decode base64/base64url data with proper padding."""
    # Convert base64url to standard base64
    data = data.replace('-', '+').replace('_', '/')
    
    # Add padding if needed
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    
    return base64.b64decode(data)


@login_required
def manage_passkeys_view(request):
    """View to manage user's passkeys."""
    passkeys = request.user.get_passkeys()
    
    context = {
        'passkeys': passkeys,
        'has_passkey': request.user.has_passkey(),
    }
    
    return render(request, 'coach/manage_passkeys.html', context)


@login_required
@require_http_methods(["POST"])
def delete_passkey_view(request, passkey_id):
    """Delete a passkey."""
    try:
        passkey = WebAuthnCredential.objects.get(
            id=passkey_id,
            user=request.user
        )
        passkey_name = passkey.name
        passkey.delete()
        messages.success(request, f'Passkey "{passkey_name}" has been deleted successfully.')
    except WebAuthnCredential.DoesNotExist:
        messages.error(request, 'Passkey not found.')
    
    return redirect('coach:manage_passkeys')


@login_required
@require_http_methods(["GET"])
def passkey_registration_begin(request):
    """Begin passkey registration by generating registration options."""
    print(f"[DEBUG] Passkey registration begin called for user: {request.user.email}")
    try:
        # Get existing credentials to exclude them
        existing_credentials = []
        for cred in request.user.get_passkeys():
            try:
                decoded_id = safe_b64decode(cred.credential_id)
                existing_credentials.append(PublicKeyCredentialDescriptor(id=decoded_id))
            except Exception as e:
                print(f"[ERROR] Failed to decode existing credential {cred.name}: {e}")
                continue
        print(f"[DEBUG] Found {len(existing_credentials)} existing credentials")
        
        rp_id = get_rp_id(request)
        origin = get_origin(request)
        print(f"[DEBUG] RP ID: {rp_id}")
        print(f"[DEBUG] Origin: {origin}")
        print(f"[DEBUG] Request host: {request.get_host()}")
        print(f"[DEBUG] Request scheme: {'https' if request.is_secure() else 'http'}")
        
        # Prepare user_id - webauthn library will handle encoding
        user_id_str = str(request.user.id)
        print(f"[DEBUG] User ID: {user_id_str}")
        print(f"[DEBUG] User name: {request.user.email}")
        print(f"[DEBUG] User display name: {request.user.first_name} {request.user.last_name}")
        
        # Generate registration options
        registration_options = generate_registration_options(
            rp_id=rp_id,
            rp_name="Verbal Coach",
            user_id=user_id_str,
            user_name=request.user.email,
            user_display_name=f"{request.user.first_name} {request.user.last_name}",
            exclude_credentials=existing_credentials if existing_credentials else None,
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
            supported_pub_key_algs=[
                COSEAlgorithmIdentifier.ECDSA_SHA_256,
                COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
            ],
        )
        
        # Store the challenge in session for verification
        request.session['passkey_challenge'] = base64.b64encode(
            registration_options.challenge
        ).decode('utf-8')
        
        # Convert to JSON and return
        return JsonResponse({
            'success': True,
            'options': json.loads(options_to_json(registration_options))
        })
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Passkey registration error: {error_details}")  # Debug logging
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


@login_required
@require_http_methods(["POST"])
def passkey_registration_complete(request):
    """Complete passkey registration by verifying the credential."""
    try:
        data = json.loads(request.body)
        credential_name = data.get('name', 'My Passkey')
        
        # Get the stored challenge
        challenge = request.session.get('passkey_challenge')
        if not challenge:
            return JsonResponse({
                'success': False,
                'error': 'No registration challenge found. Please try again.'
            }, status=400)
        
        # Get expected values for debugging
        expected_origin = get_origin(request)
        expected_rp_id = get_rp_id(request)
        expected_challenge_decoded = safe_b64decode(challenge)
        
        print(f"[DEBUG] Verifying registration:")
        print(f"[DEBUG]   Expected origin: {expected_origin}")
        print(f"[DEBUG]   Expected RP ID: {expected_rp_id}")
        print(f"[DEBUG]   Credential response ID: {data.get('id', 'N/A')}")
        
        # Verify the registration response
        verification = verify_registration_response(
            credential=data,
            expected_challenge=expected_challenge_decoded,
            expected_origin=expected_origin,
            expected_rp_id=expected_rp_id,
        )
        
        # Debug: Print available attributes
        print(f"[DEBUG] Verification object type: {type(verification)}")
        print(f"[DEBUG] Verification attributes: {[attr for attr in dir(verification) if not attr.startswith('_')]}")
        
        # Check for backup-related attributes (different versions may use different names)
        backup_eligible = False
        backup_state = False
        
        # Try different attribute names
        if hasattr(verification, 'credential_backed_up'):
            backup_eligible = getattr(verification, 'credential_backed_up', False)
        if hasattr(verification, 'credential_backup_eligible'):
            backup_eligible = getattr(verification, 'credential_backup_eligible', False)
            
        if hasattr(verification, 'credential_backup_state'):
            backup_state = getattr(verification, 'credential_backup_state', False)
        
        # Handle device_type enum if present (store as string)
        device_type = ''
        if hasattr(verification, 'credential_device_type'):
            device_type_value = getattr(verification, 'credential_device_type', None)
            print(f"[DEBUG] Device type: {device_type_value}")
            if device_type_value:
                device_type = str(device_type_value)
        
        print(f"[DEBUG] backup_eligible: {backup_eligible}, backup_state: {backup_state}, device_type: {device_type}")
        
        # Save the credential
        credential = WebAuthnCredential.objects.create(
            user=request.user,
            credential_id=base64.b64encode(verification.credential_id).decode('utf-8'),
            public_key=base64.b64encode(verification.credential_public_key).decode('utf-8'),
            sign_count=verification.sign_count,
            name=credential_name,
            aaguid=str(getattr(verification, 'aaguid', '')) if getattr(verification, 'aaguid', '') else '',
            transports=data.get('response', {}).get('transports', []),
            device_type=device_type,
            backup_eligible=backup_eligible,
            backup_state=backup_state,
        )
        
        # Clear the challenge
        del request.session['passkey_challenge']
        
        messages.success(request, f'Passkey "{credential_name}" registered successfully!')
        
        return JsonResponse({
            'success': True,
            'message': 'Passkey registered successfully!'
        })
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Passkey registration complete error: {error_details}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def passkey_authentication_begin(request):
    """Begin passkey authentication during login."""
    try:
        user_id = request.session.get('temp_user_id')
        if not user_id:
            return JsonResponse({
                'success': False,
                'error': 'Session expired. Please log in again.'
            }, status=400)
        
        user = User.objects.get(id=user_id)
        
        # Get user's credentials
        credentials = user.get_passkeys()
        if not credentials.exists():
            return JsonResponse({
                'success': False,
                'error': 'No passkeys registered for this account.'
            }, status=400)
        
        # Generate authentication options
        print(f"[DEBUG] Preparing credentials for authentication")
        allow_credentials = []
        for cred in credentials:
            try:
                decoded_id = safe_b64decode(cred.credential_id)
                allow_credentials.append(PublicKeyCredentialDescriptor(id=decoded_id))
                print(f"[DEBUG] Added credential: {cred.name}")
            except Exception as e:
                print(f"[ERROR] Failed to decode credential {cred.name}: {e}")
                continue
        
        authentication_options = generate_authentication_options(
            rp_id=get_rp_id(request),
            allow_credentials=allow_credentials,
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        
        # Store the challenge in session
        request.session['passkey_auth_challenge'] = base64.b64encode(
            authentication_options.challenge
        ).decode('utf-8')
        
        return JsonResponse({
            'success': True,
            'options': json.loads(options_to_json(authentication_options))
        })
        
    except User.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'User not found.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


@require_http_methods(["POST"])
def passkey_authentication_complete(request):
    """Complete passkey authentication and log the user in."""
    try:
        user_id = request.session.get('temp_user_id')
        if not user_id:
            return JsonResponse({
                'success': False,
                'error': 'Session expired. Please log in again.'
            }, status=400)
        
        user = User.objects.get(id=user_id)
        
        # Get the stored challenge
        challenge = request.session.get('passkey_auth_challenge')
        if not challenge:
            return JsonResponse({
                'success': False,
                'error': 'No authentication challenge found. Please try again.'
            }, status=400)
        
        data = json.loads(request.body)
        credential_id = data.get('id')
        
        print(f"[DEBUG] Looking for credential with ID: {credential_id}")
        
        # Find the credential - normalize the base64 encoding
        try:
            # The credential_id from client is base64url, decode and re-encode as standard base64
            decoded_id = safe_b64decode(credential_id)
            normalized_id = base64.b64encode(decoded_id).decode('utf-8')
            print(f"[DEBUG] Normalized credential ID: {normalized_id}")
            
            credential = WebAuthnCredential.objects.get(
                credential_id=normalized_id,
                user=user,
                is_active=True
            )
            print(f"[DEBUG] Found credential: {credential.name}")
        except WebAuthnCredential.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Credential not found.'
            }, status=400)
        
        # Verify the authentication response
        verification = verify_authentication_response(
            credential=data,
            expected_challenge=safe_b64decode(challenge),
            expected_origin=get_origin(request),
            expected_rp_id=get_rp_id(request),
            credential_public_key=safe_b64decode(credential.public_key),
            credential_current_sign_count=credential.sign_count,
        )
        
        # Update sign count and last used
        credential.sign_count = verification.new_sign_count
        credential.update_last_used()
        credential.save()
        
        # Clear session data
        del request.session['passkey_auth_challenge']
        request.session.pop('temp_user_id', None)
        
        # Log the user in
        login(request, user)
        
        return JsonResponse({
            'success': True,
            'message': 'Authentication successful!',
            'redirect_url': '/dashboard/'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def verify_passkey_view(request):
    """View for passkey verification during login."""
    user_id = request.session.get('temp_user_id')
    if not user_id:
        messages.error(request, 'Session expired. Please log in again.')
        return redirect('coach:login')
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('coach:login')
    
    if not user.has_passkey():
        messages.error(request, 'No passkeys registered for this account.')
        return redirect('coach:choose_2fa_method')
    
    context = {
        'user': user
    }
    return render(request, 'coach/verify_passkey.html', context)


# Drill Views
class DrillDetailView(DetailView):
    """Base class for drill detail views with unified ML integration, personalization, and adaptive difficulty."""
    model = Drill
    template_name = 'coach/drill_unified_base.html'
    context_object_name = 'drill'

    def get_queryset(self):
        return Drill.objects.filter(is_active=True)
    
    def get_template_names(self):
        """Dynamically select the correct template based on drill."""
        drill = self.get_object()
        drill_name = drill.name.lower()
        drill_id = drill.id
        
        # Map drill names/IDs to their specific templates
        template_map = {
            'poem echo': 'coach/drill_poem_echo.html',
            11: 'coach/drill_poem_echo.html',
            'vowel vortex': 'coach/drill_vowel_vortex.html',
            12: 'coach/drill_vowel_vortex.html',
            'mirror mimic': 'coach/drill_mirror_mimic.html',
            13: 'coach/drill_mirror_mimic.html',
            'phonetic puzzle': 'coach/drill_phonetic_puzzle.html',
            14: 'coach/drill_phonetic_puzzle.html',
            'shadow superhero': 'coach/drill_shadow_superhero.html',
            15: 'coach/drill_shadow_superhero.html',
            'pencil precision': 'coach/drill_pencil_precision.html',
            16: 'coach/drill_pencil_precision.html',
            'metronome': 'coach/drill_metronome_rhythm.html',
            21: 'coach/drill_metronome_rhythm.html',
            'pause pyramid': 'coach/drill_pause_pyramid.html',
            22: 'coach/drill_pause_pyramid.html',
            'slow-motion': 'coach/drill_slow_motion_story.html',
            23: 'coach/drill_slow_motion_story.html',
            'beat drop': 'coach/drill_beat_drop_dialogue.html',
            24: 'coach/drill_beat_drop_dialogue.html',
            'timer tag': 'coach/drill_timer_tag_team.html',
            25: 'coach/drill_timer_tag_team.html',
            'echo chamber': 'coach/drill_echo_chamber.html',
            26: 'coach/drill_echo_chamber.html',
            'filler zap': 'coach/drill_filler_zap_game.html',
            17: 'coach/drill_filler_zap_game.html',
            27: 'coach/drill_filler_zap_game.html',
            'silent switcheroo': 'coach/drill_silent_switcheroo.html',
            28: 'coach/drill_silent_switcheroo.html',
            'pause power': 'coach/drill_pause_powerup.html',
            29: 'coach/drill_pause_powerup.html',
            'filler hunt': 'coach/drill_filler_hunt_mirror.html',
            30: 'coach/drill_filler_hunt_mirror.html',
            'word swap': 'coach/drill_word_swap_whirlwind.html',
            31: 'coach/drill_word_swap_whirlwind.html',
            'echo elimination': 'coach/drill_echo_elimination.html',
            32: 'coach/drill_echo_elimination.html',
        }
        
        # Check by name first
        for key, template in template_map.items():
            if isinstance(key, str) and key in drill_name:
                return [template]
            elif isinstance(key, int) and key == drill_id:
                return [template]
        
        # Fallback to unified base
        return [self.template_name]

    def get_context_data(self, **kwargs):
        from .drill_utils import get_user_difficulty_level, get_personalized_content, get_user_weak_areas
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        # Ensure drill is in context
        context['drill'] = drill
        
        # Get user progress and personalization
        try:
            difficulty_level = get_user_difficulty_level(self.request.user, drill)
            weak_areas = get_user_weak_areas(self.request.user)
        except Exception as e:
            logger.warning(f"Error getting personalization: {e}")
            difficulty_level = 'beginner'
            weak_areas = {'filler': 0.5, 'clarity': 0.5, 'pacing': 0.5}
        
        context['difficulty_level'] = difficulty_level
        context['user_completions'] = DrillCompletion.objects.filter(
            user=self.request.user,
            drill=drill
        ).order_by('-completed_at')[:5]
        
        # Add drill-specific interactive content based on drill name
        drill_name = self.object.name.lower()
        drill_id = self.object.id
        
        # Pronunciation Drills
        if 'poem echo' in drill_name or drill_id == 11:
            context.update(self.get_poem_echo_content())
        elif 'vowel vortex' in drill_name or drill_id == 12:
            context.update(self.get_vowel_vortex_content())
        elif 'mirror mimic' in drill_name or drill_id == 13:
            context.update(self.get_mirror_mimic_content())
        elif 'phonetic puzzle' in drill_name or drill_id == 14:
            context.update(self.get_phonetic_puzzle_content())
        elif 'shadow superhero' in drill_name or drill_id == 15:
            context.update(self.get_shadow_superhero_content())
        elif 'pencil precision' in drill_name or drill_id == 16:
            context.update(self.get_pencil_precision_content())
        
        # Pacing Drills
        elif 'metronome' in drill_name or drill_id == 21:
            context.update(self.get_metronome_content())
        elif 'pause pyramid' in drill_name or drill_id == 22:
            context.update(self.get_pause_pyramid_content())
        elif 'slow-motion' in drill_name or drill_id == 23:
            context.update(self.get_slow_motion_content())
        elif 'beat drop' in drill_name or drill_id == 24:
            context.update(self.get_beat_drop_content())
        elif 'timer tag' in drill_name or drill_id == 25:
            context.update(self.get_timer_tag_content())
        elif 'echo chamber' in drill_name or drill_id == 26:
            context.update(self.get_echo_chamber_content())
        elif 'rhythm ruler' in drill_name or drill_id == 19:
            context.update(self.get_rhythm_ruler_content())
        elif 'speed controller' in drill_name or drill_id == 20:
            context.update(self.get_speed_controller_content())
        
        # Filler Words Drills
        elif 'silence master' in drill_name or drill_id == 18:
            context.update(self.get_silence_master_content())
        elif 'filler zap' in drill_name or drill_id in [17, 27]:
            context.update(self.get_filler_zap_content())
        elif 'silent switcheroo' in drill_name or drill_id == 28:
            context.update(self.get_silent_switcheroo_content())
        elif 'pause power' in drill_name or drill_id == 29:
            context.update(self.get_pause_powerup_content())
        elif 'filler hunt' in drill_name or drill_id == 30:
            context.update(self.get_filler_hunt_content())
        elif 'word swap' in drill_name or drill_id == 31:
            context.update(self.get_word_swap_content())
        elif 'echo elimination' in drill_name or drill_id == 32:
            context.update(self.get_echo_elimination_content())
        
        return context
    
    # Pronunciation Drill Content Methods
    def get_poem_echo_content(self):
        """Get content for Poem Echo Challenge drill using unified system."""
        from .drill_utils import get_personalized_content
        import json
        
        drill = self.object
        excerpts_pool = [
            {'text': "I have a dream that one day this nation will rise up and live out the true meaning of its creed."},
            {'text': "We hold these truths to be self-evident, that all men are created equal."},
            {'text': "Let freedom ring from the prodigious hilltops of New Hampshire."},
            {'text': "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin."},
            {'text': "Now is the time to make real the promises of democracy."},
            {'text': "We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead."}
        ]
        
        selected_content = get_personalized_content(drill, self.request.user, excerpts_pool)
        
        # Extract text from selected content
        if isinstance(selected_content, dict):
            mlk_excerpt = selected_content.get('text', '')
        else:
            mlk_excerpt = str(selected_content) if selected_content else ''
        
        # Fallback to random if empty
        if not mlk_excerpt:
            mlk_excerpt = self.get_random_mlk_excerpt()
        
        phonetic_tips = self.get_phonetic_tips(mlk_excerpt) if mlk_excerpt else []
        
        return {
            'drill_content_type': 'poem_echo',
            'mlk_excerpt': mlk_excerpt,
            'phonetic_tips': phonetic_tips,
            'phonetic_tips_json': json.dumps(phonetic_tips),  # For JavaScript
            'practice_text': mlk_excerpt,
            'personalization_note': selected_content.get('personalization_note', '') if isinstance(selected_content, dict) else '',
        }
    
    def get_random_mlk_excerpt(self):
        """Get a random MLK excerpt."""
        excerpts = [
            "I have a dream that one day this nation will rise up and live out the true meaning of its creed: 'We hold these truths to be self-evident, that all men are created equal.'",
            "We hold these truths to be self-evident, that all men are created equal. Let freedom ring from the prodigious hilltops of New Hampshire.",
            "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",
            "Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice.",
            "We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead. We cannot turn back.",
            "Let freedom ring from every hill and molehill of Mississippi. From every mountainside, let freedom ring."
        ]
        return random.choice(excerpts)
    
    def get_phonetic_tips(self, text):
        """Extract phonetic tips for words in the text."""
        tips_dict = {
            'rise': "Emphasize the 'r' sound (RRR-ise), make the 'i' long (EYE)",
            'dream': "Pronounce 'ea' as long 'e' (DREEM), not 'ay'",
            'nation': "Stress first syllable: 'NAY-shun', not 'nay-SHUN'",
            'freedom': "Make the 'ee' sound clear and long: 'FREE-dum'",
            'children': "Pronounce 'ch' clearly as 'CH', not 'sh'",
            'judged': "The 'dg' sounds like 'j' (JUHD), emphasize the 'd'",
            'equal': "Stress first syllable: 'EE-kwul', 'e' is long",
            'hilltops': "Two clear syllables: 'HILL-tops', stress first",
            'prodigious': "Pro-DI-jus, stress second syllable",
            'promises': "PROM-uh-siz, stress first syllable",
            'democracy': "duh-MOK-ruh-see, stress second syllable",
            'pledge': "Clear 'p' and 'l' sounds: PLEDJ",
            'mountainside': "MOUN-tin-side, three clear syllables",
        }
        
        word_tips = []
        nlp = get_spacy_model()
        
        if nlp:
            doc = nlp(text.lower())
            for token in doc:
                clean_word = token.text.strip('.,!?;:')
                if clean_word in tips_dict:
                    word_tips.append(f"'{clean_word}': {tips_dict[clean_word]}")
        else:
            # Fallback without spaCy
            for word in text.lower().split():
                clean_word = word.strip('.,!?;:')
                if clean_word in tips_dict:
                    word_tips.append(f"'{clean_word}': {tips_dict[clean_word]}")
        
        return word_tips if word_tips else ["Focus on clear enunciation", "Pronounce each word distinctly"]
    
    def get_vowel_vortex_content(self):
        """Get content for Vowel Vortex drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        vowel_words_pool = [
            {'vowel': 'A', 'words': ['apple', 'amazing', 'adventure', 'acrobat', 'astronaut']},
            {'vowel': 'E', 'words': ['elephant', 'energy', 'excellent', 'exercise', 'education']},
            {'vowel': 'I', 'words': ['incredible', 'imagination', 'important', 'interesting', 'invention']},
            {'vowel': 'O', 'words': ['outstanding', 'opportunity', 'organization', 'observation', 'optimistic']},
            {'vowel': 'U', 'words': ['unbelievable', 'understanding', 'universe', 'unusual', 'unlimited']}
        ]
        
        selected = get_personalized_content(drill, self.request.user, vowel_words_pool)
        
        if isinstance(selected, dict) and 'vowel' in selected:
            vowel_words = {selected['vowel']: selected['words']}
        else:
            vowel_words = {
                'A': ['apple', 'amazing', 'adventure', 'acrobat', 'astronaut'],
                'E': ['elephant', 'energy', 'excellent', 'exercise', 'education'],
                'I': ['incredible', 'imagination', 'important', 'interesting', 'invention'],
                'O': ['outstanding', 'opportunity', 'organization', 'observation', 'optimistic'],
                'U': ['unbelievable', 'understanding', 'universe', 'unusual', 'unlimited']
            }
        
        return {
            'drill_content_type': 'vowel_vortex',
            'vowels': ['A', 'E', 'I', 'O', 'U'],
            'vowel_words': vowel_words,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_mirror_mimic_content(self):
        """Get content for Mirror Mimic drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        twisters_pool = [
            {'text': "She sells seashells by the seashore."},
            {'text': "Peter Piper picked a peck of pickled peppers."},
            {'text': "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"},
            {'text': "Red leather, yellow leather, red leather, yellow leather."},
            {'text': "Unique New York, unique New York, you know you need unique New York."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, twisters_pool)
        
        return {
            'drill_content_type': 'mirror_mimic',
            'tongue_twisters': [t.get('text', t) if isinstance(t, dict) else t for t in twisters_pool],
            'selected_twister': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_phonetic_puzzle_content(self):
        """Get content for Phonetic Puzzle drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        words_pool = [
            {'word': 'think', 'phonetics': ['θ', 'ɪ', 'ŋ', 'k']},
            {'word': 'this', 'phonetics': ['ð', 'ɪ', 's']},
            {'word': 'ship', 'phonetics': ['ʃ', 'ɪ', 'p']},
            {'word': 'measure', 'phonetics': ['m', 'e', 'ʒ', 'ə', 'r']},
            {'word': 'church', 'phonetics': ['tʃ', 'ɜ', 'r', 'tʃ']},
            {'word': 'judge', 'phonetics': ['dʒ', 'ʌ', 'dʒ']},
            {'word': 'sing', 'phonetics': ['s', 'ɪ', 'ŋ']},
            {'word': 'red', 'phonetics': ['r', 'e', 'd']},
            {'word': 'love', 'phonetics': ['l', 'ʌ', 'v']},
            {'word': 'water', 'phonetics': ['w', 'ɔ', 't', 'ə', 'r']},
            {'word': 'yes', 'phonetics': ['j', 'e', 's']}
        ]
        
        selected = get_personalized_content(drill, self.request.user, words_pool)
        
        return {
            'drill_content_type': 'phonetic_puzzle',
            'phonetic_symbols': ['θ', 'ð', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'ŋ', 'r', 'l', 'w', 'j'],
            'target_words': words_pool,
            'selected_word': selected if isinstance(selected, dict) and 'word' in selected else random.choice(words_pool),
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_shadow_superhero_content(self):
        """Get content for Shadow Superhero drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        scripts_pool = [
            {
                'title': 'The Power of Vulnerability',
                'text': 'Vulnerability is not weakness. It is our most accurate measurement of courage.',
                'duration': 30
            },
            {
                'title': 'How Great Leaders Inspire Action',
                'text': 'People don\'t buy what you do, they buy why you do it.',
                'duration': 25
            },
            {
                'title': 'The Puzzle of Motivation',
                'text': 'The secret to high performance isn\'t rewards and punishments, but that unseen intrinsic drive.',
                'duration': 35
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, scripts_pool)
        
        return {
            'drill_content_type': 'shadow_superhero',
            'ted_scripts': scripts_pool,
            'selected_script': selected if isinstance(selected, dict) and 'title' in selected else random.choice(scripts_pool),
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_pencil_precision_content(self):
        """Get content for Pencil Precision drill."""
        import json
        precision_texts = [
            {
                'text': "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet, making it perfect for articulation practice.",
                'focus': 'All letters, clear enunciation',
                'difficulty': 'medium',
            },
            {
                'text': "She sells seashells by the seashore, and the shells she sells are seashells, I'm sure. Clear articulation is essential for effective communication.",
                'focus': 'S sounds, clarity',
                'difficulty': 'hard',
            },
            {
                'text': "How can a clam cram in a clean cream can? Precision in pronunciation requires careful attention to each consonant and vowel sound.",
                'focus': 'C, cl sounds',
                'difficulty': 'hard',
            },
            {
                'text': "A proper copper coffee pot. Practice makes perfect. Each word requires precise articulation.",
                'focus': 'P, c sounds',
                'difficulty': 'medium',
            },
            {
                'text': "Red lorry, yellow lorry, red lorry, yellow lorry. Say this clearly with the pencil between your teeth.",
                'focus': 'R, l sounds',
                'difficulty': 'hard',
            },
            {
                'text': "The art of clear speech begins with precise articulation. Every consonant and vowel must be pronounced distinctly and accurately.",
                'focus': 'General articulation',
                'difficulty': 'medium',
            },
        ]
        selected = random.choice(precision_texts)
        return {
            'drill_content_type': 'pencil_precision',
            'precision_texts': precision_texts,
            'current_text': selected,
            'current_text_json': json.dumps(selected),
        }
    
    # Pacing Drill Content Methods
    def get_metronome_content(self):
        """Get content for Metronome Rhythm Race drill."""
        from .drill_utils import get_user_difficulty_level
        
        drill = self.object
        try:
            difficulty = get_user_difficulty_level(self.request.user, drill)
            if difficulty == 'beginner':
                target_wpm = 120
                metronome_tempos = [60, 80, 100]
            elif difficulty == 'intermediate':
                target_wpm = 150
                metronome_tempos = [100, 120, 140]
            else:  # advanced
                target_wpm = 180
                metronome_tempos = [120, 140, 160]
        except:
            target_wpm = 150
            metronome_tempos = [60, 80, 100, 120, 140, 160]
        
        return {
            'drill_content_type': 'metronome',
            'metronome_tempos': metronome_tempos,
            'target_wpm': target_wpm,
            'opponent_levels': [
                {'name': 'Turtle', 'speed': 0.8, 'unlock_streak': 0},
                {'name': 'Rabbit', 'speed': 1.0, 'unlock_streak': 3},
                {'name': 'Cheetah', 'speed': 1.2, 'unlock_streak': 7},
                {'name': 'Lightning', 'speed': 1.5, 'unlock_streak': 15}
            ],
        }
    
    def get_pause_pyramid_content(self):
        """Get content for Pause Pyramid drill."""
        import json
        pyramid_texts = [
            "Effective pauses create emphasis. [PAUSE] They give listeners time to process. [PAUSE] They build anticipation. [PAUSE]",
            "Level one: Short pause. [PAUSE] Level two: Medium pause. [PAUSE] Level three: Long pause. [PAUSE] Level four: Strategic pause. [PAUSE]",
            "Pause before important points. [PAUSE] Pause after key statements. [PAUSE] Pause to create impact. [PAUSE]",
        ]
        return {
            'drill_content_type': 'pause_pyramid',
            'pyramid_levels': [
                {'level': 1, 'pause_duration': 0.5, 'text': 'Short pause (0.5s)', 'icon': '⏱️'},
                {'level': 2, 'pause_duration': 1.0, 'text': 'Medium pause (1.0s)', 'icon': '⏸️'},
                {'level': 3, 'pause_duration': 1.5, 'text': 'Long pause (1.5s)', 'icon': '⏹️'},
                {'level': 4, 'pause_duration': 2.0, 'text': 'Strategic pause (2.0s)', 'icon': '🎯'},
            ],
            'practice_text': random.choice(pyramid_texts),
            'pyramid_texts': pyramid_texts,
            'pyramid_texts_json': json.dumps(pyramid_texts),
        }
    
    def get_slow_motion_content(self):
        """Get content for Slow-Motion Story Slam drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        stories_pool = [
            {
                'title': 'The Mysterious Forest',
                'story': 'A young explorer discovers an ancient forest where time moves differently. Each step reveals new wonders.',
                'plot_twists': ['Add a talking tree!', 'Include a time portal!', 'Make it snow in summer!']
            },
            {
                'title': 'The Space Station',
                'story': 'An astronaut on a space station receives a mysterious signal from deep space.',
                'plot_twists': ['Add an alien encounter!', 'Include a malfunction!', 'Make it a dream!']
            },
            {
                'title': 'The Magic Library',
                'story': 'A librarian discovers that books in this library can transport readers to different worlds.',
                'plot_twists': ['Add a dragon!', 'Include a villain!', 'Make it interactive!']
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, stories_pool)
        
        return {
            'drill_content_type': 'slow_motion',
            'story_prompts': stories_pool,
            'selected_story': selected if isinstance(selected, dict) and 'title' in selected else random.choice(stories_pool),
            'playback_speeds': [0.5, 0.75, 1.0, 1.25, 1.5],
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_beat_drop_content(self):
        """Get content for Beat Drop Dialogue drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        prompts_pool = [
            {'text': "Tell me about your favorite hobby and why you love it."},
            {'text': "Describe a memorable vacation you took recently."},
            {'text': "Explain how you would solve world hunger."},
            {'text': "Share your thoughts on the future of technology."},
            {'text': "Discuss the importance of friendship in life."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        
        return {
            'drill_content_type': 'beat_drop',
            'beat_patterns': [
                {'name': 'Simple 4/4', 'bpm': 120, 'drops': [1, 3]},
                {'name': 'Complex 6/8', 'bpm': 140, 'drops': [1, 4]},
                {'name': 'Jazz Swing', 'bpm': 100, 'drops': [2, 4]},
            ],
            'dialogue_prompts': [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool],
            'selected_prompt': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'selected_pattern': random.choice([
                {'name': 'Simple 4/4', 'bpm': 120, 'drops': [1, 3]},
                {'name': 'Complex 6/8', 'bpm': 140, 'drops': [1, 4]},
                {'name': 'Jazz Swing', 'bpm': 100, 'drops': [2, 4]},
            ]),
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_timer_tag_content(self):
        """Get content for Timer Tag Team drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        prompts_pool = [
            {'text': "Describe your dream job and what makes it perfect."},
            {'text': "Explain how you would redesign your city."},
            {'text': "Share your favorite childhood memory."},
            {'text': "Discuss the impact of social media on society."},
            {'text': "Tell me about a skill you'd like to learn."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        
        return {
            'drill_content_type': 'timer_tag',
            'timer_cycles': [
                {'speak_time': 30, 'pause_time': 10, 'name': 'Standard'},
                {'speak_time': 45, 'pause_time': 15, 'name': 'Extended'},
                {'speak_time': 20, 'pause_time': 5, 'name': 'Quick'},
            ],
            'partner_prompts': [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool],
            'selected_cycle': random.choice([
                {'speak_time': 30, 'pause_time': 10, 'name': 'Standard'},
                {'speak_time': 45, 'pause_time': 15, 'name': 'Extended'},
                {'speak_time': 20, 'pause_time': 5, 'name': 'Quick'},
            ]),
            'selected_prompt': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_echo_chamber_content(self):
        """Get content for Echo Chamber Escalation drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        texts_pool = [
            {'text': "The quick brown fox jumps over the lazy dog."},
            {'text': "She sells seashells by the seashore."},
            {'text': "Peter Piper picked a peck of pickled peppers."},
            {'text': "How much wood would a woodchuck chuck?"},
            {'text': "Red leather, yellow leather, red leather, yellow leather."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, texts_pool)
        
        return {
            'drill_content_type': 'echo_chamber',
            'escalation_levels': [
                {'level': 1, 'speed': 1.0, 'name': 'Normal', 'description': 'Match normal speed'},
                {'level': 2, 'speed': 1.2, 'name': 'Fast', 'description': 'Match 20% faster'},
                {'level': 3, 'speed': 1.5, 'name': 'Faster', 'description': 'Match 50% faster'},
                {'level': 4, 'speed': 0.8, 'name': 'Slow', 'description': 'Match 20% slower'},
                {'level': 5, 'speed': 0.6, 'name': 'Slower', 'description': 'Match 40% slower'},
            ],
            'practice_texts': [t.get('text', t) if isinstance(t, dict) else t for t in texts_pool],
            'selected_text': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_rhythm_ruler_content(self):
        """Get content for Rhythm Ruler drill."""
        import json
        rhythm_texts = [
            "The rhythm ruler guides your speech. Follow its pattern. Maintain consistency.",
            "Pattern one: ta-TA-ta-TA. Stress alternates. Rhythm flows. Speech improves.",
            "Pattern two: TA-ta-ta-TA. Stress starts. Rhythm begins. Communication strengthens.",
        ]
        return {
            'drill_content_type': 'rhythm_ruler',
            'rhythm_patterns': [
                {'pattern': 'ta-TA-ta-TA', 'description': 'Alternating stress pattern', 'example': 'The CAT sat on the MAT', 'color': 'blue'},
                {'pattern': 'TA-ta-ta-TA', 'description': 'Starting with stress', 'example': 'BIG dog runs FAST', 'color': 'green'},
                {'pattern': 'ta-ta-TA-ta', 'description': 'Stress in the middle', 'example': 'a BIG surprise today', 'color': 'purple'},
            ],
            'practice_text': random.choice(rhythm_texts),
            'rhythm_texts': rhythm_texts,
            'rhythm_texts_json': json.dumps(rhythm_texts),
        }
    
    def get_speed_controller_content(self):
        """Get content for Speed Controller drill."""
        import json
        practice_texts = [
            "Control your speed like a conductor controls an orchestra. Each moment has its tempo. Each phrase has its pace.",
            "Start slow like a tortoise. Build speed like a walker. Reach peak like a runner. Control is everything.",
            "Speed is not about rushing. It's about precision. It's about timing. It's about control.",
        ]
        return {
            'drill_content_type': 'speed_controller',
            'speed_levels': [
                {'name': 'Tortoise', 'wpm': 100, 'description': 'Very slow, careful speech', 'icon': '🐢', 'color': 'blue'},
                {'name': 'Walking', 'wpm': 140, 'description': 'Natural, comfortable pace', 'icon': '🚶', 'color': 'green'},
                {'name': 'Running', 'wpm': 180, 'description': 'Fast, energetic delivery', 'icon': '🏃', 'color': 'red'},
            ],
            'practice_text': random.choice(practice_texts),
            'practice_texts': practice_texts,
            'practice_texts_json': json.dumps(practice_texts),
        }
    
    # Filler Words Drill Content Methods
    def get_silence_master_content(self):
        """Get content for Silence Master drill."""
        import json
        filler_challenges = [
            {
                'text': "Describe your ideal vacation without using 'um', 'uh', or 'like'. Focus on replacing fillers with intentional pauses.",
                'time_limit': 60,
                'fillers_to_avoid': ['um', 'uh', 'like', 'you know'],
                'difficulty': 'medium',
            },
            {
                'text': "Explain how to make your favorite meal without fillers. Use silence instead of hesitation words.",
                'time_limit': 90,
                'fillers_to_avoid': ['um', 'uh', 'like', 'so', 'well'],
                'difficulty': 'hard',
            },
            {
                'text': "Tell a story about a memorable event, replacing all pauses with intentional silence. No filler words allowed!",
                'time_limit': 120,
                'fillers_to_avoid': ['um', 'uh', 'like', 'you know', 'so', 'well', 'actually'],
                'difficulty': 'hard',
            },
            {
                'text': "Describe your dream job in detail without using filler words. Replace them with strategic pauses.",
                'time_limit': 75,
                'fillers_to_avoid': ['um', 'uh', 'like'],
                'difficulty': 'medium',
            },
            {
                'text': "Explain a concept you recently learned without fillers. Use silence to think, not 'um' or 'uh'.",
                'time_limit': 90,
                'fillers_to_avoid': ['um', 'uh', 'like', 'you know'],
                'difficulty': 'hard',
            },
        ]
        selected = random.choice(filler_challenges)
        return {
            'drill_content_type': 'silence_master',
            'filler_challenges': filler_challenges,
            'current_challenge': selected,
            'current_challenge_json': json.dumps(selected),
        }
    
    def get_filler_zap_content(self):
        """Get content for Filler Zap Game drill."""
        import json
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically']
        topics = [
            {
                'text': "Your favorite book or movie",
                'difficulty': 'easy',
                'time_limit': 60,
            },
            {
                'text': "A skill you want to learn",
                'difficulty': 'medium',
                'time_limit': 75,
            },
            {
                'text': "A place you'd like to visit",
                'difficulty': 'easy',
                'time_limit': 60,
            },
            {
                'text': "Your dream job",
                'difficulty': 'medium',
                'time_limit': 75,
            },
            {
                'text': "A memorable experience from your childhood",
                'difficulty': 'hard',
                'time_limit': 90,
            },
        ]
        selected_topic = random.choice(topics)
        target_fillers = random.sample(filler_words, 3)
        return {
            'drill_content_type': 'filler_zap',
            'filler_words': filler_words,
            'target_fillers': target_fillers,
            'target_fillers_json': json.dumps(target_fillers),
            'topics': topics,
            'current_topic': selected_topic,
            'current_topic_json': json.dumps(selected_topic),
        }
    
    def get_silent_switcheroo_content(self):
        """Get content for Silent Switcheroo drill."""
        from .drill_utils import get_personalized_content
        from coach.utils import detect_filler_words
        
        drill = self.object
        sentences_pool = [
            {'text': "Um, I think, like, it's great and, you know, really amazing."},
            {'text': "So, basically, the thing is, uh, that we need to, like, figure this out."},
            {'text': "Well, actually, I believe that, um, the solution is, you know, quite simple."},
            {'text': "Like, honestly, I feel that, uh, this approach is, so, much better."},
            {'text': "You know, the problem is that, um, we haven't really, like, considered this."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, sentences_pool)
        selected_sentence = selected.get('text', selected) if isinstance(selected, dict) else selected
        
        filler_analysis = detect_filler_words(selected_sentence)
        detected_fillers = [filler['word'] for filler in filler_analysis['fillers']]
        
        return {
            'drill_content_type': 'silent_switcheroo',
            'original_sentence': selected_sentence,
            'detected_fillers': detected_fillers,
            'filler_analysis': filler_analysis,
            'replacement_suggestions': [
                "Use a pause instead of 'um' or 'uh'",
                "Replace 'like' with more specific words",
                "Remove 'you know' and be more direct",
                "Substitute 'so' with 'therefore' or 'thus'",
                "Replace 'well' with 'however' or 'moreover'"
            ],
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_pause_powerup_content(self):
        """Get content for Pause Power-Up drill."""
        from .drill_utils import get_personalized_content, get_user_difficulty_level
        
        drill = self.object
        try:
            difficulty = get_user_difficulty_level(self.request.user, drill)
            if difficulty == 'beginner':
                power_levels = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                ]
            elif difficulty == 'intermediate':
                power_levels = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                    {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                ]
            else:
                power_levels = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                    {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                    {'level': 4, 'duration': 90, 'name': 'Super', 'color': 'orange'},
                    {'level': 5, 'duration': 120, 'name': 'Ultimate', 'color': 'red'},
                ]
        except:
            power_levels = [
                {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                {'level': 4, 'duration': 90, 'name': 'Super', 'color': 'orange'},
                {'level': 5, 'duration': 120, 'name': 'Ultimate', 'color': 'red'},
            ]
        
        topics_pool = [
            {'text': "Describe your ideal vacation destination"},
            {'text': "Explain how technology has changed your life"},
            {'text': "Share your favorite book and why you recommend it"},
            {'text': "Discuss the importance of exercise and health"},
            {'text': "Tell me about a challenge you overcame"}
        ]
        
        selected = get_personalized_content(drill, self.request.user, topics_pool)
        
        return {
            'drill_content_type': 'pause_powerup',
            'power_levels': power_levels,
            'speech_topics': [t.get('text', t) if isinstance(t, dict) else t for t in topics_pool],
            'selected_topic': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_filler_hunt_content(self):
        """Get content for Filler Hunt Mirror Maze drill."""
        from .drill_utils import get_personalized_content
        from coach.utils import detect_filler_words
        
        drill = self.object
        scenarios_pool = [
            {
                'title': 'Job Interview',
                'text': 'Tell me about yourself and your qualifications.',
                'common_fillers': ['um', 'uh', 'like', 'you know']
            },
            {
                'title': 'Presentation',
                'text': 'Present your ideas for improving our company.',
                'common_fillers': ['so', 'well', 'basically', 'actually']
            },
            {
                'title': 'Casual Conversation',
                'text': 'Describe your weekend plans to a friend.',
                'common_fillers': ['like', 'um', 'uh', 'you know']
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, scenarios_pool)
        selected_scenario = selected if isinstance(selected, dict) and 'title' in selected else random.choice(scenarios_pool)
        
        prompt = selected_scenario.get('text', selected_scenario.get('prompt', ''))
        filler_analysis = detect_filler_words(prompt)
        
        return {
            'drill_content_type': 'filler_hunt',
            'hunt_scenarios': scenarios_pool,
            'selected_scenario': selected_scenario,
            'speech_challenge': prompt,
            'speech_challenges': [s.get('text', s.get('prompt', '')) for s in scenarios_pool],
            'filler_analysis': filler_analysis,
            'hunt_tips': [
                "Pause instead of saying 'um' or 'uh'",
                "Use 'however' instead of 'but'",
                "Replace 'like' with 'such as' or 'for example'",
                "Say 'in addition' instead of 'also'",
                "Use 'therefore' instead of 'so'"
            ],
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_word_swap_content(self):
        """Get content for Word Swap Whirlwind drill."""
        from .drill_utils import get_personalized_content
        
        drill = self.object
        topics_pool = [
            {'text': "Explain how to make a perfect cup of coffee"},
            {'text': "Describe your dream house in detail"},
            {'text': "Share your thoughts on artificial intelligence"},
            {'text': "Tell me about your favorite season and why"},
            {'text': "Discuss the impact of social media on relationships"}
        ]
        
        selected = get_personalized_content(drill, self.request.user, topics_pool)
        
        return {
            'drill_content_type': 'word_swap',
            'filler_replacements': [
                {'filler': 'um', 'replacement': 'pause', 'category': 'Hesitation'},
                {'filler': 'uh', 'replacement': 'silence', 'category': 'Hesitation'},
                {'filler': 'like', 'replacement': 'such as', 'category': 'Comparison'},
                {'filler': 'you know', 'replacement': 'as you understand', 'category': 'Acknowledgment'},
                {'filler': 'so', 'replacement': 'therefore', 'category': 'Conclusion'},
                {'filler': 'well', 'replacement': 'however', 'category': 'Transition'},
                {'filler': 'actually', 'replacement': 'in fact', 'category': 'Emphasis'},
                {'filler': 'basically', 'replacement': 'essentially', 'category': 'Simplification'},
            ],
            'impromptu_topics': [t.get('text', t) if isinstance(t, dict) else t for t in topics_pool],
            'selected_topic': selected.get('text', selected) if isinstance(selected, dict) else selected,
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }
    
    def get_echo_elimination_content(self):
        """Get content for Echo Elimination Echo drill."""
        from .drill_utils import get_personalized_content
        from coach.utils import detect_filler_words
        
        drill = self.object
        texts_pool = [
            {'text': "Um, I think that, like, the solution is, you know, quite simple."},
            {'text': "So, basically, what I'm saying is that, uh, we need to, like, work together."},
            {'text': "Well, actually, the thing is that, um, this approach is, you know, better."},
            {'text': "Like, honestly, I feel that, uh, this method is, so, much more effective."},
            {'text': "You know, the problem is that, um, we haven't really, like, considered this."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, texts_pool)
        selected_text = selected.get('text', selected) if isinstance(selected, dict) else selected
        
        filler_analysis = detect_filler_words(selected_text)
        
        return {
            'drill_content_type': 'echo_elimination',
            'practice_texts': [t.get('text', t) if isinstance(t, dict) else t for t in texts_pool],
            'selected_text': selected_text,
            'filler_analysis': filler_analysis,
            'detected_fillers': [filler['word'] for filler in filler_analysis['fillers']],
            'elimination_levels': [
                {'level': 1, 'name': 'Easy', 'fillers_removed': 2, 'description': 'Remove 2 fillers'},
                {'level': 2, 'name': 'Medium', 'fillers_removed': 4, 'description': 'Remove 4 fillers'},
                {'level': 3, 'name': 'Hard', 'fillers_removed': 6, 'description': 'Remove 6 fillers'},
            ],
            'personalization_note': selected.get('personalization_note', '') if isinstance(selected, dict) else '',
        }


class PoemEchoChallengeView(DrillDetailView):
    """Poem Echo Challenge drill with unified ML integration."""
    template_name = 'coach/drill_poem_echo.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        # Get personalized content
        excerpts_pool = [
            {'text': "I have a dream that one day this nation will rise up and live out the true meaning of its creed."},
            {'text': "We hold these truths to be self-evident, that all men are created equal."},
            {'text': "Let freedom ring from the prodigious hilltops of New Hampshire."},
            {'text': "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin."},
            {'text': "Now is the time to make real the promises of democracy."},
            {'text': "We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead."}
        ]
        
        selected_content = get_personalized_content(drill, self.request.user, excerpts_pool)
        
        # Extract text from selected content
        if isinstance(selected_content, dict):
            mlk_excerpt = selected_content.get('text', '')
        else:
            mlk_excerpt = str(selected_content) if selected_content else ''
        
        # Fallback to random if empty
        if not mlk_excerpt:
            mlk_excerpt = self.get_random_mlk_excerpt()
        
        context['mlk_excerpt'] = mlk_excerpt
        context['phonetic_tips'] = self.get_phonetic_tips(mlk_excerpt) if mlk_excerpt else []
        context['personalization_note'] = selected_content.get('personalization_note', '') if isinstance(selected_content, dict) else ''
        
        # Debug: Log to ensure data is being passed
        logger.info(f"Poem Echo - Excerpt: {mlk_excerpt[:50]}..., Tips count: {len(context['phonetic_tips'])}")
        
        return context
    
    def get_random_mlk_excerpt(self):
        excerpts = [
            "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
            "We hold these truths to be self-evident, that all men are created equal.",
            "Let freedom ring from the prodigious hilltops of New Hampshire.",
            "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin.",
            "Now is the time to make real the promises of democracy.",
            "We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead."
        ]
        return random.choice(excerpts)
    
    def get_phonetic_tips(self, text):
        tips = {
            'rise': "Emphasize the 'r' sound and make the 'i' long",
            'dream': "Pronounce the 'ea' as a long 'e' sound",
            'nation': "Stress the first syllable 'NA-tion'",
            'freedom': "Make the 'ee' sound clear and long",
            'children': "Pronounce 'ch' clearly, not 'sh'",
            'judged': "The 'dg' should sound like 'j' not 'g'"
        }
        
        word_tips = []
        nlp = get_spacy_model()
        
        if nlp:
            doc = nlp(text.lower())
            for token in doc:
                clean_word = token.text.strip('.,!?')
                if clean_word in tips:
                    word_tips.append(f"'{clean_word}': {tips[clean_word]}")
        else:
            # Fallback without spaCy
            for word in text.lower().split():
                clean_word = word.strip('.,!?')
                if clean_word in tips:
                    word_tips.append(f"'{clean_word}': {tips[clean_word]}")
        
        return word_tips


class ExaggeratedVowelVortexView(DrillDetailView):
    """Exaggerated Vowel Vortex drill with unified ML integration."""
    template_name = 'coach/drill_vowel_vortex.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['vowels'] = ['A', 'E', 'I', 'O', 'U']
        vowel_words_pool = [
            {'vowel': 'A', 'words': ['apple', 'amazing', 'adventure', 'acrobat', 'astronaut']},
            {'vowel': 'E', 'words': ['elephant', 'energy', 'excellent', 'exercise', 'education']},
            {'vowel': 'I', 'words': ['incredible', 'imagination', 'important', 'interesting', 'invention']},
            {'vowel': 'O', 'words': ['outstanding', 'opportunity', 'organization', 'observation', 'optimistic']},
            {'vowel': 'U', 'words': ['unbelievable', 'understanding', 'universe', 'unusual', 'unlimited']}
        ]
        
        selected = get_personalized_content(drill, self.request.user, vowel_words_pool)
        if isinstance(selected, dict) and 'vowel' in selected:
            context['vowel_words'] = {selected['vowel']: selected['words']}
        else:
            context['vowel_words'] = {
                'A': ['apple', 'amazing', 'adventure', 'acrobat', 'astronaut'],
                'E': ['elephant', 'energy', 'excellent', 'exercise', 'education'],
                'I': ['incredible', 'imagination', 'important', 'interesting', 'invention'],
                'O': ['outstanding', 'opportunity', 'organization', 'observation', 'optimistic'],
                'U': ['unbelievable', 'understanding', 'universe', 'unusual', 'unlimited']
            }
        
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class MirrorMimicMadnessView(DrillDetailView):
    """Mirror Mimic Madness drill with unified ML integration."""
    template_name = 'coach/drill_mirror_mimic.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        twisters_pool = [
            {'text': "She sells seashells by the seashore."},
            {'text': "Peter Piper picked a peck of pickled peppers."},
            {'text': "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"},
            {'text': "Red leather, yellow leather, red leather, yellow leather."},
            {'text': "Unique New York, unique New York, you know you need unique New York."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, twisters_pool)
        context['tongue_twisters'] = [t.get('text', t) if isinstance(t, dict) else t for t in twisters_pool]
        context['selected_twister'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class PhoneticPuzzleBuilderView(DrillDetailView):
    """Phonetic Puzzle Builder drill with unified ML integration."""
    template_name = 'coach/drill_phonetic_puzzle.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['phonetic_symbols'] = ['θ', 'ð', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'ŋ', 'r', 'l', 'w', 'j']
        words_pool = [
            {'word': 'think', 'phonetics': ['θ', 'ɪ', 'ŋ', 'k']},
            {'word': 'this', 'phonetics': ['ð', 'ɪ', 's']},
            {'word': 'ship', 'phonetics': ['ʃ', 'ɪ', 'p']},
            {'word': 'measure', 'phonetics': ['m', 'e', 'ʒ', 'ə', 'r']},
            {'word': 'church', 'phonetics': ['tʃ', 'ɜ', 'r', 'tʃ']},
            {'word': 'judge', 'phonetics': ['dʒ', 'ʌ', 'dʒ']},
            {'word': 'sing', 'phonetics': ['s', 'ɪ', 'ŋ']},
            {'word': 'red', 'phonetics': ['r', 'e', 'd']},
            {'word': 'love', 'phonetics': ['l', 'ʌ', 'v']},
            {'word': 'water', 'phonetics': ['w', 'ɔ', 't', 'ə', 'r']},
            {'word': 'yes', 'phonetics': ['j', 'e', 's']}
        ]
        
        selected = get_personalized_content(drill, self.request.user, words_pool)
        context['target_words'] = words_pool
        context['selected_word'] = selected if isinstance(selected, dict) and 'word' in selected else random.choice(words_pool)
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        
        # Add spaCy model for verification
        nlp = get_spacy_model()
        if nlp:
            context['spacy_available'] = True
            doc = nlp(context['selected_word']['word'])
            context['word_tokens'] = [token.text for token in doc]
        else:
            context['spacy_available'] = False
            
        return context


class ShadowSuperheroView(DrillDetailView):
    """Shadow Superhero drill with unified ML integration."""
    template_name = 'coach/drill_shadow_superhero.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        scripts_pool = [
            {
                'title': 'The Power of Vulnerability',
                'text': 'Vulnerability is not weakness. It is our most accurate measurement of courage.',
                'duration': 30
            },
            {
                'title': 'How Great Leaders Inspire Action',
                'text': 'People don\'t buy what you do, they buy why you do it.',
                'duration': 25
            },
            {
                'title': 'The Puzzle of Motivation',
                'text': 'The secret to high performance isn\'t rewards and punishments, but that unseen intrinsic drive.',
                'duration': 35
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, scripts_pool)
        context['ted_scripts'] = scripts_pool
        context['selected_script'] = selected if isinstance(selected, dict) and 'title' in selected else random.choice(scripts_pool)
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class PencilPrecisionDrillView(DrillDetailView):
    """Pencil Precision Drill with unified ML integration."""
    template_name = 'coach/drill_pencil_precision.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        texts_pool = [
            {'text': "The quick brown fox jumps over the lazy dog."},
            {'text': "She sells seashells by the seashore."},
            {'text': "Peter Piper picked a peck of pickled peppers."},
            {'text': "How much wood would a woodchuck chuck?"},
            {'text': "Red leather, yellow leather, red leather, yellow leather."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, texts_pool)
        context['precision_texts'] = [t.get('text', t) if isinstance(t, dict) else t for t in texts_pool]
        context['selected_text'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


# Pacing Drills
class MetronomeRhythmRaceView(DrillDetailView):
    """Metronome Rhythm Race drill with unified ML integration."""
    template_name = 'coach/drill_metronome_rhythm.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_user_difficulty_level
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        try:
            difficulty = get_user_difficulty_level(self.request.user, drill)
            if difficulty == 'beginner':
                context['target_wpm'] = 120
                context['metronome_tempos'] = [60, 80, 100]
            elif difficulty == 'intermediate':
                context['target_wpm'] = 150
                context['metronome_tempos'] = [100, 120, 140]
            else:  # advanced
                context['target_wpm'] = 180
                context['metronome_tempos'] = [120, 140, 160]
        except:
            context['target_wpm'] = 150
            context['metronome_tempos'] = [60, 80, 100, 120, 140, 160]
        
        context['opponent_levels'] = [
            {'name': 'Turtle', 'speed': 0.8, 'unlock_streak': 0},
            {'name': 'Rabbit', 'speed': 1.0, 'unlock_streak': 3},
            {'name': 'Cheetah', 'speed': 1.2, 'unlock_streak': 7},
            {'name': 'Lightning', 'speed': 1.5, 'unlock_streak': 15}
        ]
        return context


class PausePyramidBuilderView(DrillDetailView):
    """Pause Pyramid Builder drill with unified ML integration."""
    template_name = 'coach/drill_pause_pyramid.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['pyramid_levels'] = [
            {'level': 1, 'pause_duration': 1, 'description': '1-second pause'},
            {'level': 2, 'pause_duration': 2, 'description': '2-second pause'},
            {'level': 3, 'pause_duration': 3, 'description': '3-second pause'},
            {'level': 4, 'pause_duration': 4, 'description': '4-second pause'},
            {'level': 5, 'pause_duration': 5, 'description': '5-second pause'},
        ]
        
        prompts_pool = [
            {'text': "The future of technology is bright. We must embrace change."},
            {'text': "Education is the foundation of society. It shapes our tomorrow."},
            {'text': "Climate change affects us all. We need immediate action."},
            {'text': "Innovation drives progress. Creativity fuels advancement."},
            {'text': "Leadership requires courage. Vision guides the way."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        context['speech_prompts'] = [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool]
        context['selected_prompt'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class SlowMotionStorySlamView(DrillDetailView):
    """Slow-Motion Story Slam drill with unified ML integration."""
    template_name = 'coach/drill_slow_motion_story.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        stories_pool = [
            {
                'title': 'The Mysterious Forest',
                'story': 'A young explorer discovers an ancient forest where time moves differently. Each step reveals new wonders.',
                'plot_twists': ['Add a talking tree!', 'Include a time portal!', 'Make it snow in summer!']
            },
            {
                'title': 'The Space Station',
                'story': 'An astronaut on a space station receives a mysterious signal from deep space.',
                'plot_twists': ['Add an alien encounter!', 'Include a malfunction!', 'Make it a dream!']
            },
            {
                'title': 'The Magic Library',
                'story': 'A librarian discovers that books in this library can transport readers to different worlds.',
                'plot_twists': ['Add a dragon!', 'Include a villain!', 'Make it interactive!']
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, stories_pool)
        context['story_prompts'] = stories_pool
        context['selected_story'] = selected if isinstance(selected, dict) and 'title' in selected else random.choice(stories_pool)
        context['playback_speeds'] = [0.5, 0.75, 1.0, 1.25, 1.5]
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class BeatDropDialogueView(DrillDetailView):
    """Beat Drop Dialogue drill with unified ML integration."""
    template_name = 'coach/drill_beat_drop_dialogue.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['beat_patterns'] = [
            {'name': 'Simple 4/4', 'bpm': 120, 'drops': [1, 3]},
            {'name': 'Complex 6/8', 'bpm': 140, 'drops': [1, 4]},
            {'name': 'Jazz Swing', 'bpm': 100, 'drops': [2, 4]},
        ]
        
        prompts_pool = [
            {'text': "Tell me about your favorite hobby and why you love it."},
            {'text': "Describe a memorable vacation you took recently."},
            {'text': "Explain how you would solve world hunger."},
            {'text': "Share your thoughts on the future of technology."},
            {'text': "Discuss the importance of friendship in life."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        context['dialogue_prompts'] = [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool]
        context['selected_prompt'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['selected_pattern'] = random.choice(context['beat_patterns'])
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class TimerTagTeamView(DrillDetailView):
    """Timer Tag Team drill with unified ML integration."""
    template_name = 'coach/drill_timer_tag_team.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['timer_cycles'] = [
            {'speak_time': 30, 'pause_time': 10, 'name': 'Standard'},
            {'speak_time': 45, 'pause_time': 15, 'name': 'Extended'},
            {'speak_time': 20, 'pause_time': 5, 'name': 'Quick'},
        ]
        
        prompts_pool = [
            {'text': "Describe your dream job and what makes it perfect."},
            {'text': "Explain how you would redesign your city."},
            {'text': "Share your favorite childhood memory."},
            {'text': "Discuss the impact of social media on society."},
            {'text': "Tell me about a skill you'd like to learn."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        context['partner_prompts'] = [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool]
        context['selected_cycle'] = random.choice(context['timer_cycles'])
        context['selected_prompt'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class EchoChamberEscalationView(DrillDetailView):
    """Echo Chamber Escalation drill with unified ML integration."""
    template_name = 'coach/drill_echo_chamber.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['escalation_levels'] = [
            {'level': 1, 'speed': 1.0, 'name': 'Normal', 'description': 'Match normal speed'},
            {'level': 2, 'speed': 1.2, 'name': 'Fast', 'description': 'Match 20% faster'},
            {'level': 3, 'speed': 1.5, 'name': 'Faster', 'description': 'Match 50% faster'},
            {'level': 4, 'speed': 0.8, 'name': 'Slow', 'description': 'Match 20% slower'},
            {'level': 5, 'speed': 0.6, 'name': 'Slower', 'description': 'Match 40% slower'},
        ]
        
        texts_pool = [
            {'text': "The quick brown fox jumps over the lazy dog."},
            {'text': "She sells seashells by the seashore."},
            {'text': "Peter Piper picked a peck of pickled peppers."},
            {'text': "How much wood would a woodchuck chuck?"},
            {'text': "Red leather, yellow leather, red leather, yellow leather."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, texts_pool)
        context['practice_texts'] = [t.get('text', t) if isinstance(t, dict) else t for t in texts_pool]
        context['selected_text'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


# Filler Words Drills
class FillerZapGameView(DrillDetailView):
    """Filler Zap Game drill with unified ML integration."""
    template_name = 'coach/drill_filler_zap_game.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content, get_user_difficulty_level
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['filler_words'] = ['um', 'uh', 'like', 'you know', 'so', 'well', 'actually', 'basically']
        
        try:
            difficulty = get_user_difficulty_level(self.request.user, drill)
            if difficulty == 'beginner':
                context['game_levels'] = [
                    {'level': 1, 'time_limit': 60, 'target_score': 50, 'name': 'Beginner'},
                ]
            elif difficulty == 'intermediate':
                context['game_levels'] = [
                    {'level': 1, 'time_limit': 60, 'target_score': 50, 'name': 'Beginner'},
                    {'level': 2, 'time_limit': 45, 'target_score': 75, 'name': 'Intermediate'},
                ]
            else:
                context['game_levels'] = [
                    {'level': 1, 'time_limit': 60, 'target_score': 50, 'name': 'Beginner'},
                    {'level': 2, 'time_limit': 45, 'target_score': 75, 'name': 'Intermediate'},
                    {'level': 3, 'time_limit': 30, 'target_score': 100, 'name': 'Advanced'},
                ]
        except:
            context['game_levels'] = [
                {'level': 1, 'time_limit': 60, 'target_score': 50, 'name': 'Beginner'},
                {'level': 2, 'time_limit': 45, 'target_score': 75, 'name': 'Intermediate'},
                {'level': 3, 'time_limit': 30, 'target_score': 100, 'name': 'Advanced'},
            ]
        
        prompts_pool = [
            {'text': "Tell me about your favorite movie and why you like it."},
            {'text': "Describe what you did last weekend in detail."},
            {'text': "Explain how to make your favorite dish."},
            {'text': "Share your thoughts on the weather today."},
            {'text': "Discuss your plans for next week."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, prompts_pool)
        context['speech_prompts'] = [p.get('text', p) if isinstance(p, dict) else p for p in prompts_pool]
        context['selected_prompt'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class SilentSwitcherooView(DrillDetailView):
    """Silent Switcheroo drill with unified ML integration."""
    template_name = 'coach/drill_silent_switcheroo.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        sentences_pool = [
            {'text': "Um, I think, like, it's great and, you know, really amazing."},
            {'text': "So, basically, the thing is, uh, that we need to, like, figure this out."},
            {'text': "Well, actually, I believe that, um, the solution is, you know, quite simple."},
            {'text': "Like, honestly, I feel that, uh, this approach is, so, much better."},
            {'text': "You know, the problem is that, um, we haven't really, like, considered this."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, sentences_pool)
        selected_sentence = selected.get('text', selected) if isinstance(selected, dict) else selected
        
        # Use filler detection to analyze the selected sentence
        filler_analysis = detect_filler_words(selected_sentence)
        detected_fillers = [filler['word'] for filler in filler_analysis['fillers']]
        
        context['original_sentence'] = selected_sentence
        context['detected_fillers'] = detected_fillers
        context['filler_analysis'] = filler_analysis
        context['replacement_suggestions'] = [
            "Use a pause instead of 'um' or 'uh'",
            "Replace 'like' with more specific words",
            "Remove 'you know' and be more direct",
            "Substitute 'so' with 'therefore' or 'thus'",
            "Replace 'well' with 'however' or 'moreover'"
        ]
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class PausePowerUpView(DrillDetailView):
    """Pause Power-Up drill with unified ML integration."""
    template_name = 'coach/drill_pause_powerup.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content, get_user_difficulty_level
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        try:
            difficulty = get_user_difficulty_level(self.request.user, drill)
            if difficulty == 'beginner':
                context['power_levels'] = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                ]
            elif difficulty == 'intermediate':
                context['power_levels'] = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                    {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                ]
            else:
                context['power_levels'] = [
                    {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                    {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                    {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                    {'level': 4, 'duration': 90, 'name': 'Super', 'color': 'orange'},
                    {'level': 5, 'duration': 120, 'name': 'Ultimate', 'color': 'red'},
                ]
        except:
            context['power_levels'] = [
                {'level': 1, 'duration': 30, 'name': 'Starter', 'color': 'green'},
                {'level': 2, 'duration': 45, 'name': 'Charger', 'color': 'blue'},
                {'level': 3, 'duration': 60, 'name': 'Power', 'color': 'purple'},
                {'level': 4, 'duration': 90, 'name': 'Super', 'color': 'orange'},
                {'level': 5, 'duration': 120, 'name': 'Ultimate', 'color': 'red'},
            ]
        
        topics_pool = [
            {'text': "Describe your ideal vacation destination"},
            {'text': "Explain how technology has changed your life"},
            {'text': "Share your favorite book and why you recommend it"},
            {'text': "Discuss the importance of exercise and health"},
            {'text': "Tell me about a challenge you overcame"}
        ]
        
        selected = get_personalized_content(drill, self.request.user, topics_pool)
        context['speech_topics'] = [t.get('text', t) if isinstance(t, dict) else t for t in topics_pool]
        context['selected_topic'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class FillerHuntMirrorMazeView(DrillDetailView):
    """Filler Hunt Mirror Maze drill with unified ML integration."""
    template_name = 'coach/drill_filler_hunt_mirror.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        scenarios_pool = [
            {
                'title': 'Job Interview',
                'text': 'Tell me about yourself and your qualifications.',
                'common_fillers': ['um', 'uh', 'like', 'you know']
            },
            {
                'title': 'Presentation',
                'text': 'Present your ideas for improving our company.',
                'common_fillers': ['so', 'well', 'basically', 'actually']
            },
            {
                'title': 'Casual Conversation',
                'text': 'Describe your weekend plans to a friend.',
                'common_fillers': ['like', 'um', 'uh', 'you know']
            }
        ]
        
        selected = get_personalized_content(drill, self.request.user, scenarios_pool)
        context['hunt_scenarios'] = scenarios_pool
        context['selected_scenario'] = selected if isinstance(selected, dict) and 'title' in selected else random.choice(scenarios_pool)
        
        # Use filler detection to analyze the selected prompt
        prompt = context['selected_scenario'].get('text', context['selected_scenario'].get('prompt', ''))
        filler_analysis = detect_filler_words(prompt)
        
        context['speech_challenge'] = prompt
        context['speech_challenges'] = [s.get('text', s.get('prompt', '')) for s in scenarios_pool]
        context['filler_analysis'] = filler_analysis
        context['hunt_tips'] = [
            "Pause instead of saying 'um' or 'uh'",
            "Use 'however' instead of 'but'",
            "Replace 'like' with 'such as' or 'for example'",
            "Say 'in addition' instead of 'also'",
            "Use 'therefore' instead of 'so'"
        ]
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class WordSwapWhirlwindView(DrillDetailView):
    """Word Swap Whirlwind drill with unified ML integration."""
    template_name = 'coach/drill_word_swap_whirlwind.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        context['filler_replacements'] = [
            {'filler': 'um', 'replacement': 'pause', 'category': 'Hesitation'},
            {'filler': 'uh', 'replacement': 'silence', 'category': 'Hesitation'},
            {'filler': 'like', 'replacement': 'such as', 'category': 'Comparison'},
            {'filler': 'you know', 'replacement': 'as you understand', 'category': 'Acknowledgment'},
            {'filler': 'so', 'replacement': 'therefore', 'category': 'Conclusion'},
            {'filler': 'well', 'replacement': 'however', 'category': 'Transition'},
            {'filler': 'actually', 'replacement': 'in fact', 'category': 'Emphasis'},
            {'filler': 'basically', 'replacement': 'essentially', 'category': 'Simplification'},
        ]
        
        topics_pool = [
            {'text': "Explain how to make a perfect cup of coffee"},
            {'text': "Describe your dream house in detail"},
            {'text': "Share your thoughts on artificial intelligence"},
            {'text': "Tell me about your favorite season and why"},
            {'text': "Discuss the impact of social media on relationships"}
        ]
        
        selected = get_personalized_content(drill, self.request.user, topics_pool)
        context['impromptu_topics'] = [t.get('text', t) if isinstance(t, dict) else t for t in topics_pool]
        context['selected_topic'] = selected.get('text', selected) if isinstance(selected, dict) else selected
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


class EchoEliminationEchoView(DrillDetailView):
    """Echo Elimination Echo drill with unified ML integration."""
    template_name = 'coach/drill_echo_elimination.html'
    
    def get_context_data(self, **kwargs):
        from .drill_utils import get_personalized_content
        
        context = super().get_context_data(**kwargs)
        drill = self.object
        
        texts_pool = [
            {'text': "Um, I think that, like, the solution is, you know, quite simple."},
            {'text': "So, basically, what I'm saying is that, uh, we need to, like, work together."},
            {'text': "Well, actually, the thing is that, um, this approach is, you know, better."},
            {'text': "Like, honestly, I feel that, uh, this method is, so, much more effective."},
            {'text': "You know, the problem is that, um, we haven't really, like, considered this."}
        ]
        
        selected = get_personalized_content(drill, self.request.user, texts_pool)
        selected_text = selected.get('text', selected) if isinstance(selected, dict) else selected
        
        context['practice_texts'] = [t.get('text', t) if isinstance(t, dict) else t for t in texts_pool]
        context['selected_text'] = selected_text
        
        # Use filler detection to analyze the selected text
        filler_analysis = detect_filler_words(selected_text)
        
        context['filler_analysis'] = filler_analysis
        context['detected_fillers'] = [filler['word'] for filler in filler_analysis['fillers']]
        context['elimination_levels'] = [
            {'level': 1, 'name': 'Easy', 'fillers_removed': 2, 'description': 'Remove 2 fillers'},
            {'level': 2, 'name': 'Medium', 'fillers_removed': 4, 'description': 'Remove 4 fillers'},
            {'level': 3, 'name': 'Hard', 'fillers_removed': 6, 'description': 'Remove 6 fillers'},
        ]
        context['personalization_note'] = selected.get('personalization_note', '') if isinstance(selected, dict) else ''
        return context


@login_required
def drills_list_view(request):
    """Display all available drills organized by skill type."""
    drills = Drill.objects.filter(is_active=True).order_by('skill_type', 'name')
    
    # Group drills by skill type
    drills_by_type = {}
    for drill in drills:
        skill_type = drill.get_skill_type_display()
        if skill_type not in drills_by_type:
            drills_by_type[skill_type] = []
        drills_by_type[skill_type].append(drill)
    
    context = {
        'drills_by_type': drills_by_type,
        'total_drills': drills.count(),
    }
    
    return render(request, 'coach/drills_list.html', context)


@login_required
@require_http_methods(["POST"])
def complete_drill(request):
    """Unified drill completion endpoint using ML-based scoring."""
    from .drill_utils import save_drill_completion
    import json
    
    try:
        drill_id = request.POST.get('drill_id')
        if not drill_id:
            return JsonResponse({'error': 'drill_id is required'}, status=400)
        
        drill = get_object_or_404(Drill, id=drill_id, is_active=True)
        
        # Get ML scores and metrics from request
        ml_scores = json.loads(request.POST.get('ml_scores', '{}'))
        ml_metrics = json.loads(request.POST.get('ml_metrics', '{}'))
        drill_specific_metrics = json.loads(request.POST.get('drill_specific_metrics', '{}'))
        difficulty_level = request.POST.get('difficulty_level', 'beginner')
        duration_seconds = int(request.POST.get('duration_seconds', 0))
        notes = request.POST.get('notes', '')
        
        # Get audio file if provided
        audio_data = None
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
        
        # Create analysis result dict
        analysis_result = {
            'ml_scores': ml_scores,
            'ml_metrics': ml_metrics,
            'drill_specific_metrics': drill_specific_metrics,
            'duration': duration_seconds
        }
        
        # Save completion using unified utility
        completion = save_drill_completion(
            user=request.user,
            drill=drill,
            analysis_result=analysis_result,
            audio_data=audio_data,
            difficulty_level=difficulty_level,
            notes=notes
        )
        
        return JsonResponse({
            'success': True,
            'completion_id': completion.id,
            'overall_score': completion.get_overall_score(),
            'message': 'Drill completed successfully!'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"Error completing drill: {e}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)

# Real-time Speech Analysis API Endpoints
@login_required
@require_http_methods(["POST"])
def analyze_speech_api(request):
    """Enhanced API endpoint to analyze speech and save feedback with drill recommendation."""
    try:
        import base64
        from io import BytesIO
        import tempfile
        import os
        from django.core.files.base import ContentFile
        
        # Get audio data from request
        audio_data = request.POST.get('audio')
        if not audio_data:
            return JsonResponse({'error': 'No audio data provided'}, status=400)
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Transcribe audio
            transcription_result = transcribe_audio(tmp_path)
            
            if not transcription_result['success']:
                return JsonResponse({
                    'error': 'Transcription failed',
                    'details': transcription_result.get('error')
                }, status=500)
            
            text = transcription_result['text']
            duration = float(request.POST.get('duration', 0))
            
            # GPU-accelerated analysis pipeline with caching
            pipeline_result = analyze_speech_pipeline(
                audio_path=tmp_path,
                transcript=text,
                duration_seconds=duration
            )

            if not pipeline_result.get('success'):
                error_message = pipeline_result.get('error', 'Analysis failed')
                detected_language = pipeline_result.get('language')

                if detected_language and detected_language.lower() not in {'en', 'en-us', 'en-gb'}:
                    return JsonResponse({
                        'error': error_message,
                        'details': f"Detected language '{detected_language}'. Please upload or record English speech."
                    }, status=400)

                return JsonResponse({'error': error_message}, status=500)

            analysis = pipeline_result.get('quality', {})
            bert_scores = pipeline_result.get('bert_scores', {})
            cause_prediction = pipeline_result.get('cause_prediction', {'cause': 'other', 'confidence': 0.0})
            
            # Save audio file permanently
            audio_filename = f'voices/user_{request.user.id}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.wav'
            audio_file = default_storage.save(
                audio_filename,
                ContentFile(audio_bytes, name='recording.wav')
            )
            
            # Get or calculate basic metrics
            filler_data = analysis.get('filler_analysis', {})
            filler_count = filler_data.get('count', 0)
            filler_density = filler_data.get('density', 0.0)
            wpm = analysis.get('wpm', 0)
            f1_score = filler_density  # Legacy metric
            
            # Create Feedback object
            feedback = Feedback.objects.create(
                user=request.user,
                f1_score=f1_score,
                fillers_count=filler_count,
                wpm=int(wpm),
                transcription=text,
                audio_file=audio_file,
                filler_score=bert_scores.get('filler', 0.0),
                clarity_score=bert_scores.get('clarity', 0.0),
                pacing_score=bert_scores.get('pacing', 0.0),
                cause=cause_prediction.get('cause', 'other')
            )
            
            # Recommend drill
            # Pass actual metrics to help cross-validate model scores
            bert_scores_with_metrics = bert_scores.copy()
            # Calculate WPM and filler metrics from transcript and duration
            word_count = len(text.split())
            duration_minutes = duration / 60.0 if duration > 0 else 0
            wpm = int((word_count / duration_minutes) if duration_minutes > 0 else 0)
            filler_count = len([w for w in text.lower().split() if w in ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well']])
            filler_density = (filler_count / word_count * 100) if word_count > 0 else 0
            
            bert_scores_with_metrics['_actual_metrics'] = {
                'wpm': wpm,
                'filler_count': filler_count,
                'filler_density': filler_density
            }
            
            recommender = DrillRecommender()
            history = list(Feedback.objects.filter(user=request.user).order_by('-created_at')[:7])
            recommended_drill = recommender.pick_best(bert_scores_with_metrics, history)
            
            if recommended_drill:
                feedback.recommended_drill = recommended_drill
                feedback.save()
            
            # Get growth data
            growth_data = Feedback.objects.filter(
                user=request.user,
                created_at__gte=timezone.now() - timedelta(days=7)
            ).order_by('created_at')
            
            growth_chart = {
                'dates': [f.created_at.strftime('%Y-%m-%d %H:%M') for f in growth_data],
                'filler_scores': [f.filler_score if f.filler_score else 0 for f in growth_data],
                'clarity_scores': [f.clarity_score if f.clarity_score else 0 for f in growth_data],
                'pacing_scores': [f.pacing_score if f.pacing_score else 0 for f in growth_data],
            }
            
            return JsonResponse({
                'success': True,
                'transcription': text,
                'analysis': analysis,
                'feedback': {
                    'id': feedback.id,
                    'filler_score': feedback.filler_score,
                    'clarity_score': feedback.clarity_score,
                    'pacing_score': feedback.pacing_score,
                    'cause': feedback.cause,
                },
                'drill': {
                    'id': recommended_drill.id if recommended_drill else None,
                    'name': recommended_drill.name if recommended_drill else None,
                    'url': f'/coach/drill/{recommended_drill.id}/' if recommended_drill else None,
                },
                'growth': growth_chart
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error in speech analysis: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def analyze_speech_async_upload(request):
    """
    Accept an uploaded audio file and trigger asynchronous analysis via Celery.
    """
    audio = request.FILES.get('vn') or request.FILES.get('audio')
    if audio is None:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    duration_raw = request.POST.get('duration')
    try:
        duration = float(duration_raw) if duration_raw else None
    except (TypeError, ValueError):
        duration = None

    secure_name = f"user_{request.user.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}_{audio.name}"
    saved_path = default_storage.save(f"voices/async/{secure_name}", audio)
    absolute_path = default_storage.path(saved_path)

    task = analyze_speech_async.delay(
        absolute_path,
        transcript=None,
        duration_seconds=duration,
        force_refresh=False,
    )
    return JsonResponse({'task_id': task.id, 'status': 'queued'})


@login_required
@require_http_methods(["GET"])
def analyze_speech_task_status(request, task_id):
    """
    Poll the status of an asynchronous speech analysis task.
    """
    result = AsyncResult(task_id)

    if result.successful():
        return JsonResponse({'status': 'done', 'result': result.result})
    if result.failed():
        info = result.info
        if isinstance(info, dict):
            error_message = info.get('error', str(info))
            detected_language = info.get('language')
            if detected_language and detected_language.lower() not in {'en', 'en-us', 'en-gb'}:
                return JsonResponse({
                    'status': 'failed',
                    'error': error_message,
                    'details': f"Detected language '{detected_language}'. Please upload or record English speech."
                }, status=400)
        else:
            error_message = str(info)
        return JsonResponse({'status': 'failed', 'error': error_message}, status=500)
    return JsonResponse({'status': result.status.lower()})


@login_required
@require_http_methods(["POST"])
def detect_fillers_api(request):
    """API endpoint to detect filler words in real-time."""
    try:
        text = request.POST.get('text', '')
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        filler_analysis = detect_filler_words(text)
        
        return JsonResponse({
            'success': True,
            'fillers': filler_analysis['fillers'],
            'count': filler_analysis['count'],
            'density': filler_analysis['density']
        })
        
    except Exception as e:
        logger.error(f"Error detecting fillers: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def analyze_drill_api(request):
    """Unified API endpoint to analyze drill practice using ML models."""
    from .drill_utils import analyze_drill_audio
    import tempfile
    import os
    
    try:
        # Get audio file from request
        if 'audio' not in request.FILES:
            return JsonResponse({'error': 'No audio file provided'}, status=400)
        
        audio_file = request.FILES['audio']
        drill_id = request.POST.get('drill_id')
        
        if not drill_id:
            return JsonResponse({'success': False, 'error': 'drill_id is required'}, status=400)
        
        try:
            drill = Drill.objects.get(id=int(drill_id), is_active=True)
        except Drill.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Drill not found'}, status=404)
        except (ValueError, TypeError):
            return JsonResponse({'success': False, 'error': 'Invalid drill_id'}, status=400)
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Get drill-specific parameters
        drill_specific_params = {
            'target_text': request.POST.get('target_text'),
            'target_wpm': int(request.POST.get('target_wpm', 150)),
            'phonetic_targets': request.POST.get('phonetic_targets'),
            'baseline_fillers': int(request.POST.get('baseline_fillers', 0)),
            'duration': float(request.POST.get('duration', 0))
        }
        
        # Analyze using unified utility
        analysis_result = analyze_drill_audio(
            audio_data=audio_data,
            drill=drill,
            user=request.user,
            drill_specific_params=drill_specific_params
        )
        
        if not analysis_result.get('success'):
            return JsonResponse({
                'error': analysis_result.get('error', 'Analysis failed'),
                'details': analysis_result.get('details')
            }, status=500)
        
        # Clean up temp file if exists
        if 'audio_path' in analysis_result:
            try:
                if os.path.exists(analysis_result['audio_path']):
                    os.unlink(analysis_result['audio_path'])
            except Exception:
                pass
        
        # Return unified response format
        return JsonResponse({
            'success': True,
            'transcription': analysis_result.get('transcription', ''),
            'duration': analysis_result.get('duration', 0),
            'ml_scores': analysis_result.get('ml_scores', {}),
            'ml_metrics': analysis_result.get('ml_metrics', {}),
            'drill_specific_metrics': analysis_result.get('drill_specific_metrics', {}),
            'cause': analysis_result.get('cause', 'other'),
            'cause_confidence': analysis_result.get('cause_confidence', 0.0),
            'recommendations': analysis_result.get('recommendations', [])
        })
        
    except Exception as e:
        import traceback
        logger.error(f"Error in analyze_drill_api: {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_http_methods(["GET"])
def drill_progress_api(request):
    """API endpoint to get user's drill progress."""
    from .drill_utils import get_user_progress
    
    try:
        drill_id = request.GET.get('drill_id')
        drill = None
        if drill_id:
            try:
                drill = Drill.objects.get(id=int(drill_id), is_active=True)
            except Drill.DoesNotExist:
                return JsonResponse({'error': 'Drill not found'}, status=404)
        
        progress = get_user_progress(request.user, drill)
        
        return JsonResponse({
            'success': True,
            'progress': progress
        })
        
    except Exception as e:
        logger.error(f"Error getting drill progress: {e}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


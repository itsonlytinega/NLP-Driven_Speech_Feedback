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
from django.urls import reverse_lazy
from django.utils import timezone
from django_otp.plugins.otp_totp.models import TOTPDevice
from django_otp.decorators import otp_required
from django_otp import login as otp_login
import qrcode
import qrcode.image.svg
import io
import base64
from .forms import CustomUserCreationForm, CustomAuthenticationForm, CustomPasswordResetForm, TwoFactorChoiceForm, OTPVerificationForm
from .models import User, Feedback, Drill, EmailOTP
from .utils import send_verification_email, send_otp_email, create_email_otp, verify_email_token, send_password_reset_email


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
    """Display user dashboard with feedback summary."""
    # Get recent feedback for the user
    recent_feedback = Feedback.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Calculate average metrics
    feedbacks = Feedback.objects.filter(user=request.user)
    avg_f1_score = feedbacks.aggregate(avg_f1=models.Avg('f1_score'))['avg_f1'] or 0
    avg_fillers = feedbacks.aggregate(avg_fillers=models.Avg('fillers_count'))['avg_fillers'] or 0
    avg_wpm = feedbacks.aggregate(avg_wpm=models.Avg('wpm'))['avg_wpm'] or 0
    
    # Get available drills
    drills = Drill.objects.filter(is_active=True).order_by('type', 'difficulty_level')
    
    context = {
        'recent_feedback': recent_feedback,
        'avg_f1_score': round(avg_f1_score, 2),
        'avg_fillers': round(avg_fillers, 1),
        'avg_wpm': round(avg_wpm, 1),
        'total_sessions': feedbacks.count(),
        'drills': drills,
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
def feedback_view(request, feedback_id=None):
    """Display detailed feedback analysis."""
    if feedback_id:
        try:
            feedback = Feedback.objects.get(id=feedback_id, user=request.user)
            context = {'feedback': feedback}
            return render(request, 'coach/feedback.html', context)
        except Feedback.DoesNotExist:
            messages.error(request, 'Feedback not found.')
            return redirect('coach:dashboard')
    
    # Show all feedback for the user
    feedbacks = Feedback.objects.filter(user=request.user).order_by('-created_at')
    context = {'feedbacks': feedbacks}
    return render(request, 'coach/feedback.html', context)


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

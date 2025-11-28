"""
Utility functions for email operations and verification.
"""
import secrets
import string
from datetime import timedelta
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.core.signing import TimestampSigner
from django.utils import timezone
from django.conf import settings
from .models import User, EmailOTP


def send_verification_email(user):
    """
    Send email verification link to user.
    """
    signer = TimestampSigner()
    token = signer.sign(user.email)
    verification_link = f"http://127.0.0.1:8000/verify-email/{token}/"
    
    subject = 'Verify Your Email - Verbal Coach'
    context = {
        'user': user,
        'verification_link': verification_link,
        'site_name': 'Verbal Coach'
    }
    
    # Render both HTML and text versions
    html_message = render_to_string('emails/verification_email.html', context)
    text_message = render_to_string('emails/verification_email.txt', context)
    
    try:
        send_mail(
            subject=subject,
            message=text_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f"Failed to send verification email: {e}")
        return False


def send_otp_email(user, otp_code):
    """
    Send OTP code to user's email for 2FA.
    """
    subject = 'Your 2FA Code - Verbal Coach'
    context = {
        'user': user,
        'otp_code': otp_code,
        'site_name': 'Verbal Coach'
    }
    
    # Render both HTML and text versions
    html_message = render_to_string('emails/otp_email.html', context)
    text_message = render_to_string('emails/otp_email.txt', context)
    
    try:
        send_mail(
            subject=subject,
            message=text_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f"Failed to send OTP email: {e}")
        return False


def generate_otp_code(length=6):
    """
    Generate a random OTP code.
    """
    return ''.join(secrets.choice(string.digits) for _ in range(length))


def create_email_otp(user, expires_minutes=5):
    """
    Create a new email OTP for the user.
    """
    # Delete any existing unused OTPs for this user
    EmailOTP.objects.filter(user=user, is_used=False).delete()
    
    # Generate new OTP code
    code = generate_otp_code()
    expires_at = timezone.now() + timedelta(minutes=expires_minutes)
    
    # Create OTP record
    otp = EmailOTP.objects.create(
        user=user,
        code=code,
        expires_at=expires_at
    )
    
    return otp


def verify_email_token(token):
    """
    Verify email verification token and return user if valid.
    """
    try:
        signer = TimestampSigner()
        email = signer.unsign(token, max_age=settings.EMAIL_VERIFICATION_TIMEOUT)
        user = User.objects.get(email=email)
        return user
    except Exception:
        return None


def send_password_reset_email(user, reset_token):
    """
    Send password reset email to user.
    """
    reset_link = f"http://127.0.0.1:8000/reset/{reset_token}/"
    
    subject = 'Password Reset - Verbal Coach'
    context = {
        'user': user,
        'reset_link': reset_link,
        'site_name': 'Verbal Coach'
    }
    
    # Render both HTML and text versions
    html_message = render_to_string('emails/password_reset.html', context)
    text_message = render_to_string('emails/password_reset.txt', context)
    
    try:
        send_mail(
            subject=subject,
            message=text_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f"Failed to send password reset email: {e}")
        return False

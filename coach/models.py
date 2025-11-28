from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django_otp.plugins.otp_totp.models import TOTPDevice
from phonenumber_field.modelfields import PhoneNumberField
import json


class UserManager(BaseUserManager):
    """Custom user manager for email-based authentication."""
    
    def create_user(self, email, password=None, **extra_fields):
        """Create and return a regular user with an email and password."""
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        """Create and return a superuser with an email and password."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Custom User model extending AbstractUser."""
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    
    # Email verification field
    is_email_verified = models.BooleanField(
        default=False,
        help_text="Whether the user's email address has been verified"
    )
    
    # 2FA fields
    is_2fa_enabled = models.BooleanField(
        default=False,
        help_text="Whether 2FA is enabled for this user"
    )
    phone_number = PhoneNumberField(
        blank=True,
        null=True,
        help_text="Phone number for SMS fallback (optional)"
    )
    backup_codes = models.JSONField(
        default=list,
        blank=True,
        help_text="Backup codes for 2FA recovery"
    )
    
    # Remove username field since we're using email
    username = None
    
    objects = UserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"
    
    def get_totp_device(self):
        """Get the user's TOTP device if it exists."""
        return TOTPDevice.objects.filter(user=self, confirmed=True).first()
    
    def has_2fa_setup(self):
        """Check if user has 2FA properly set up."""
        return self.is_2fa_enabled and self.get_totp_device() is not None
    
    def generate_backup_codes(self, count=10):
        """Generate backup codes for 2FA recovery."""
        import secrets
        import string
        
        codes = []
        for _ in range(count):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(code)
        
        self.backup_codes = codes
        self.save()
        return codes
    
    def has_passkey(self):
        """Check if user has at least one registered passkey."""
        return self.webauthn_credentials.filter(is_active=True).exists()
    
    def get_passkeys(self):
        """Get all active passkeys for the user."""
        return self.webauthn_credentials.filter(is_active=True)


class Feedback(models.Model):
    """Model to store user feedback and analysis results."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    f1_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="F1-score for disfluency analysis (0.0 to 1.0)"
    )
    fillers_count = models.PositiveIntegerField(
        help_text="Number of filler words detected"
    )
    wpm = models.PositiveIntegerField(
        help_text="Words per minute speaking rate"
    )
    transcription = models.TextField(
        blank=True,
        help_text="Transcribed audio text"
    )
    audio_file = models.FileField(
        upload_to='audio_uploads/',
        blank=True,
        null=True,
        help_text="Uploaded audio file"
    )
    
    # Ensemble model scores (NEW)
    filler_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Filler score from ensemble model (0.0-1.0, higher = more fillers)"
    )
    clarity_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Clarity score from ensemble model (0.0-1.0, higher = worse clarity)"
    )
    pacing_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Pacing score from ensemble model (0.0-1.0, higher = worse pacing)"
    )
    
    # Cause classification (NEW)
    CAUSE_CHOICES = [
        ('lexical', 'Lexical'),
        ('syntactic', 'Syntactic'),
        ('articulatory', 'Articulatory'),
        ('fluency', 'Fluency'),
        ('other', 'Other'),
    ]
    cause = models.CharField(
        max_length=20,
        choices=CAUSE_CHOICES,
        blank=True,
        null=True,
        help_text="Predicted cause of speech issue"
    )
    
    # Recommended drill
    recommended_drill = models.ForeignKey(
        'Drill',
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='recommended_feedbacks',
        help_text="AI-recommended drill based on analysis"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for {self.user.email} - F1: {self.f1_score:.2f}"
    
    def get_growth_data(self, days=7):
        """Get growth data for the last N days."""
        from django.utils import timezone
        from datetime import timedelta
        
        cutoff_date = timezone.now() - timedelta(days=days)
        recent = Feedback.objects.filter(
            user=self.user,
            created_at__gte=cutoff_date
        ).order_by('created_at')
        
        return {
            'dates': [f.created_at.strftime('%Y-%m-%d') for f in recent],
            'filler_scores': [f.filler_score if f.filler_score else 0 for f in recent],
            'clarity_scores': [f.clarity_score if f.clarity_score else 0 for f in recent],
            'pacing_scores': [f.pacing_score if f.pacing_score else 0 for f in recent],
        }
    
    def improvement_percentage(self):
        """Calculate improvement percentage over the last 7 sessions."""
        recent = Feedback.objects.filter(
            user=self.user
        ).order_by('-created_at')[:7]
        
        if len(recent) < 2:
            return None
        
        # Calculate average fillers for first 3 vs last 3
        first_half = recent[len(recent)//2:]
        second_half = recent[:len(recent)//2]
        
        if not first_half or not second_half:
            return None
        
        avg_first = sum(f.fillers_count for f in first_half) / len(first_half)
        avg_second = sum(f.fillers_count for f in second_half) / len(second_half)
        
        if avg_first == 0:
            return 0
        
        improvement = ((avg_first - avg_second) / avg_first) * 100
        return round(improvement, 1)


class Drill(models.Model):
    """Model to store different types of speaking drills."""
    SKILL_TYPE_CHOICES = [
        ('pronunciation', 'Pronunciation'),
        ('pacing', 'Pacing'),
        ('filler_words', 'Filler Words'),
    ]
    
    CAUSE_CHOICES = [
        ('anxiety', 'Anxiety'),
        ('stress', 'Stress'),
        ('lack_of_confidence', 'Lack of Confidence'),
        ('poor_skills', 'Poor Skills'),
    ]
    
    name = models.CharField(max_length=100, default="Untitled Drill", help_text="Name of the drill")
    skill_type = models.CharField(
        max_length=20, 
        choices=SKILL_TYPE_CHOICES,
        default='pronunciation',
        help_text="Type of skill this drill targets"
    )
    cause = models.CharField(
        max_length=20, 
        choices=CAUSE_CHOICES,
        blank=True,
        null=True,
        help_text="Root cause this drill addresses (optional)"
    )
    description = models.TextField(default="", help_text="Detailed description of the drill")
    interactive_elements = models.JSONField(
        default=dict,
        help_text="Interactive elements configuration (e.g., timer_duration, prompts)"
    )
    reference_audio = models.FileField(
        upload_to='drill_reference_audio/%Y/%m/%d/',
        blank=True,
        null=True,
        help_text="Reference audio file for pronunciation drills (optional)"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['skill_type', 'name']
    
    def __str__(self):
        return f"{self.name} ({self.get_skill_type_display()})"


class DrillCompletion(models.Model):
    """Model to track user drill completions and progress with ML-based metrics."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='drill_completions')
    drill = models.ForeignKey(Drill, on_delete=models.CASCADE, related_name='completions')
    completed_at = models.DateTimeField(auto_now_add=True)
    
    # Legacy self-rated score (kept for backward compatibility)
    score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)],
        help_text="User's self-rated score (0.0-10.0) - deprecated, use ml_scores instead"
    )
    
    # ML-based objective scores (0.0-1.0, higher = better performance)
    ml_scores = models.JSONField(
        default=dict,
        blank=True,
        help_text="ML-based objective scores: {filler_score, clarity_score, pacing_score, overall_score}"
    )
    
    # Detailed ML metrics
    ml_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed ML metrics: {fillers_count, filler_density, wpm, transcription, etc.}"
    )
    
    # Drill-specific metrics
    drill_specific_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Drill-specific metrics (e.g., pronunciation_accuracy, pause_quality, rhythm_score)"
    )
    
    # Difficulty level used
    difficulty_level = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        help_text="Difficulty level used for this completion (beginner, intermediate, advanced)"
    )
    
    # Audio file reference (if stored)
    audio_file = models.FileField(
        upload_to='drill_recordings/%Y/%m/%d/',
        blank=True,
        null=True,
        help_text="Recorded audio file for this drill completion"
    )
    
    notes = models.TextField(
        blank=True,
        help_text="User's notes about the drill session"
    )
    duration_seconds = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="Time spent on the drill in seconds"
    )
    
    class Meta:
        ordering = ['-completed_at']
        unique_together = ['user', 'drill', 'completed_at']
    
    def __str__(self):
        return f"{self.user.email} - {self.drill.name} ({self.completed_at.strftime('%Y-%m-%d')})"
    
    def get_overall_score(self):
        """Get overall ML score (0.0-1.0, higher = better)."""
        if self.ml_scores and 'overall_score' in self.ml_scores:
            return self.ml_scores['overall_score']
        # Fallback: calculate from individual scores
        scores = self.ml_scores or {}
        filler = scores.get('filler_score', 0.5)
        clarity = scores.get('clarity_score', 0.5)
        pacing = scores.get('pacing_score', 0.5)
        # Convert to 0-1 scale where higher is better (invert filler since it's currently higher=worse)
        filler_normalized = 1.0 - min(1.0, filler)  # Invert filler score
        clarity_normalized = 1.0 - min(1.0, clarity)  # Invert clarity score
        pacing_normalized = 1.0 - min(1.0, pacing)  # Invert pacing score
        return (filler_normalized + clarity_normalized + pacing_normalized) / 3.0


class EmailOTP(models.Model):
    """Model to store temporary email OTP codes for 2FA."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='email_otps')
    code = models.CharField(max_length=6, help_text="6-digit OTP code")
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(help_text="When this OTP code expires")
    is_used = models.BooleanField(default=False, help_text="Whether this code has been used")
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"EmailOTP for {self.user.email} - {self.code[:2]}****"
    
    def is_expired(self):
        """Check if the OTP code has expired."""
        from django.utils import timezone
        return timezone.now() > self.expires_at
    
    def is_valid(self):
        """Check if the OTP code is valid (not used and not expired)."""
        return not self.is_used and not self.is_expired()


class WebAuthnCredential(models.Model):
    """Model to store WebAuthn (passkey) credentials for users."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='webauthn_credentials')
    
    # Credential identification
    credential_id = models.TextField(
        unique=True,
        help_text="Base64-encoded credential ID from WebAuthn"
    )
    name = models.CharField(
        max_length=100,
        help_text="User-friendly name for this credential (e.g., 'MacBook Pro Touch ID')"
    )
    
    # Credential data
    public_key = models.TextField(
        help_text="Base64-encoded public key"
    )
    sign_count = models.PositiveIntegerField(
        default=0,
        help_text="Signature counter to prevent replay attacks"
    )
    
    # Credential metadata
    aaguid = models.CharField(
        max_length=100,
        blank=True,
        help_text="Authenticator Attestation GUID"
    )
    credential_type = models.CharField(
        max_length=50,
        default='public-key',
        help_text="Type of credential"
    )
    transports = models.JSONField(
        default=list,
        blank=True,
        help_text="Transport methods supported (e.g., ['usb', 'nfc', 'ble', 'internal'])"
    )
    
    # Device information
    device_type = models.CharField(
        max_length=50,
        blank=True,
        help_text="Type of authenticator device"
    )
    backup_eligible = models.BooleanField(
        default=False,
        help_text="Whether the credential is backup eligible"
    )
    backup_state = models.BooleanField(
        default=False,
        help_text="Whether the credential is currently backed up"
    )
    
    # Status and tracking
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this credential is active and can be used"
    )
    last_used = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last time this credential was used for authentication"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'WebAuthn Credential'
        verbose_name_plural = 'WebAuthn Credentials'
    
    def __str__(self):
        return f"{self.name} - {self.user.email}"
    
    def update_last_used(self):
        """Update the last_used timestamp."""
        from django.utils import timezone
        self.last_used = timezone.now()
        self.save(update_fields=['last_used'])


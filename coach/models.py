from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django_otp.plugins.otp_totp.models import TOTPDevice
from phonenumber_field.modelfields import PhoneNumberField


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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for {self.user.email} - F1: {self.f1_score:.2f}"


class Drill(models.Model):
    """Model to store different types of speaking drills."""
    DRILL_TYPES = [
        ('breathing', 'Breathing Exercises'),
        ('pacing', 'Pacing Practice'),
        ('articulation', 'Articulation Drills'),
        ('confidence', 'Confidence Building'),
        ('presentation', 'Presentation Skills'),
    ]
    
    type = models.CharField(max_length=20, choices=DRILL_TYPES)
    title = models.CharField(max_length=100)
    instruction = models.TextField()
    duration_minutes = models.PositiveIntegerField(
        default=5,
        help_text="Recommended duration in minutes"
    )
    difficulty_level = models.PositiveIntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Difficulty level from 1 (beginner) to 5 (expert)"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['type', 'difficulty_level']
    
    def __str__(self):
        return f"{self.get_type_display()}: {self.title}"


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


from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator

User = get_user_model()


class SpeechSession(models.Model):
    """
    Model to store speech session records with analysis data.
    
    This model tracks user speech sessions including duration, filler count,
    pacing analysis, and processing status for the verbal coaching system.
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('analyzed', 'Analyzed'),
        ('archived', 'Archived'),
    ]
    
    # Core fields
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='speech_sessions',
        help_text="User who owns this speech session"
    )
    date = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time when the session was created"
    )
    duration = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="Duration of the speech session in seconds"
    )
    filler_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of filler words detected (e.g., 'um', 'uh', 'like')"
    )
    pacing_analysis = models.TextField(
        blank=True,
        help_text="Analysis of speaking pace and rhythm patterns"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Processing status of the speech session"
    )
    
    # Additional metadata
    audio_file = models.FileField(
        upload_to='speech_sessions/%Y/%m/%d/',
        blank=True,
        null=True,
        help_text="Original audio file for the speech session"
    )
    transcription = models.TextField(
        blank=True,
        help_text="Transcribed text from the speech session"
    )
    confidence_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Overall confidence score for the analysis (0.0-1.0)"
    )
    
    # ML Model Scores (0.0-1.0, higher = worse performance)
    filler_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Filler score from ML model (0.0-1.0, higher = more fillers)"
    )
    clarity_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Clarity score from ML model (0.0-1.0, higher = worse clarity)"
    )
    pacing_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Pacing score from ML model (0.0-1.0, higher = worse pacing)"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Speech Session'
        verbose_name_plural = 'Speech Sessions'
        indexes = [
            models.Index(fields=['user', '-date']),
            models.Index(fields=['status']),
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"Session {self.id} - {self.user.email} ({self.date.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def duration_minutes(self):
        """Return duration in minutes as a formatted string."""
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes}:{seconds:02d}"
    
    @property
    def filler_rate(self):
        """Calculate filler words per minute."""
        if self.duration == 0:
            return 0
        return (self.filler_count / self.duration) * 60
    
    def get_status_display_class(self):
        """Return CSS class for status display."""
        status_classes = {
            'pending': 'bg-yellow-100 text-yellow-800',
            'analyzed': 'bg-green-100 text-green-800',
            'archived': 'bg-gray-100 text-gray-800',
        }
        return status_classes.get(self.status, 'bg-gray-100 text-gray-800')


class DetectedFiller(models.Model):
    """
    Model to store individual filler words detected in speech transcripts.
    
    This model stores each detected filler word with its position, timing,
    and context for detailed analysis and feedback.
    """
    
    # Foreign key to speech session
    speech_session = models.ForeignKey(
        SpeechSession,
        on_delete=models.CASCADE,
        related_name='detected_fillers',
        help_text="Speech session this filler was detected in"
    )
    
    # Filler word details
    filler_word = models.CharField(
        max_length=50,
        help_text="The detected filler word (e.g., 'um', 'uh', 'like')"
    )
    
    # Position in transcript
    word_position = models.IntegerField(
        help_text="Position of the filler word in the transcript (0-indexed)"
    )
    
    # Timing information (if available from transcription)
    start_time = models.FloatField(
        blank=True,
        null=True,
        help_text="Start time of the filler word in seconds"
    )
    end_time = models.FloatField(
        blank=True,
        null=True,
        help_text="End time of the filler word in seconds"
    )
    
    # Context (surrounding words)
    context_before = models.CharField(
        max_length=100,
        blank=True,
        help_text="Words before the filler (for context)"
    )
    context_after = models.CharField(
        max_length=100,
        blank=True,
        help_text="Words after the filler (for context)"
    )
    
    # Detection method
    detection_method = models.CharField(
        max_length=20,
        choices=[
            ('spacy', 'spaCy NLP'),
            ('regex', 'Regex Pattern'),
            ('bert', 'BERT Model'),
            ('rule_based', 'Rule-based'),
        ],
        default='rule_based',
        help_text="Method used to detect this filler word"
    )
    
    # Confidence score (0.0-1.0)
    confidence = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Confidence score for this detection (0.0-1.0)"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['speech_session', 'word_position']
        verbose_name = 'Detected Filler'
        verbose_name_plural = 'Detected Fillers'
        indexes = [
            models.Index(fields=['speech_session', 'word_position']),
            models.Index(fields=['filler_word']),
            models.Index(fields=['detection_method']),
        ]
    
    def __str__(self):
        return f"{self.filler_word} at position {self.word_position} in session {self.speech_session.id}"
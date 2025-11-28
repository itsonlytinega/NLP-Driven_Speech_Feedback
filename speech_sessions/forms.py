from django import forms
from django.core.validators import FileExtensionValidator
from .models import SpeechSession


class SpeechSessionCreateForm(forms.ModelForm):
    """Form for creating new speech sessions."""
    
    class Meta:
        model = SpeechSession
        fields = ['audio_file', 'transcription']  # Removed duration - will be auto-calculated
        widgets = {
            'audio_file': forms.FileInput(attrs={
                'class': 'mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100',
                'accept': '.mp3,.wav,.m4a,.ogg,.flac',
            }),
            'transcription': forms.Textarea(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'rows': 4,
                'placeholder': 'Optional: Provide a transcription of your speech...',
            }),
        }
        help_texts = {
            'audio_file': 'Upload an audio file or record directly from your browser. Duration will be automatically calculated.',
            'transcription': 'Optional: You can provide a manual transcription or leave this blank for automatic transcription.',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make audio_file required (either upload or record)
        self.fields['audio_file'].required = False  # Will be validated in clean method
        # Add file extension validator for audio files
        self.fields['audio_file'].validators.append(
            FileExtensionValidator(
                allowed_extensions=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
                message='Please upload a valid audio file (MP3, WAV, M4A, OGG, or FLAC).'
            )
        )
    
    def clean(self):
        """Validate that either audio_file or recorded_audio is provided."""
        cleaned_data = super().clean()
        # Note: We'll handle recorded audio in the view, not in form validation
        # This allows flexibility for both upload and recording
        return cleaned_data


class SpeechSessionUpdateForm(forms.ModelForm):
    """Form for updating existing speech sessions."""
    
    class Meta:
        model = SpeechSession
        fields = ['duration', 'filler_count', 'pacing_analysis', 'status', 'transcription', 'confidence_score']
        widgets = {
            'duration': forms.NumberInput(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'min': '1',
            }),
            'filler_count': forms.NumberInput(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'min': '0',
            }),
            'pacing_analysis': forms.Textarea(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'rows': 4,
                'placeholder': 'Analysis of speaking pace and rhythm patterns...',
            }),
            'status': forms.Select(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
            }),
            'transcription': forms.Textarea(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'rows': 6,
            }),
            'confidence_score': forms.NumberInput(attrs={
                'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
                'step': '0.01',
                'min': '0',
                'max': '1',
            }),
        }
    
    def clean_confidence_score(self):
        """Validate confidence score is between 0 and 1."""
        score = self.cleaned_data.get('confidence_score')
        if score is not None and (score < 0 or score > 1):
            raise forms.ValidationError('Confidence score must be between 0.0 and 1.0.')
        return score


class SpeechSessionFilterForm(forms.Form):
    """Form for filtering speech sessions."""
    
    STATUS_CHOICES = [('', 'All Statuses')] + SpeechSession.STATUS_CHOICES
    
    status = forms.ChoiceField(
        choices=STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
        })
    )
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
            'type': 'date',
        }),
        help_text='Filter sessions from this date'
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
            'type': 'date',
        }),
        help_text='Filter sessions up to this date'
    )
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm',
            'placeholder': 'Search transcriptions...',
        }),
        help_text='Search in transcriptions and pacing analysis'
    )


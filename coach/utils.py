"""
Utility functions for email operations and verification.
"""
from __future__ import annotations
import secrets
import string
import re
import logging
import os
import hashlib
import threading
from datetime import timedelta
from functools import lru_cache

from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.core.signing import TimestampSigner
from django.utils import timezone
from django.conf import settings
from django.urls import reverse
from django.core.cache import cache

from .models import User, EmailOTP

try:
    import torch
except ImportError:
    torch = None

# Setup logging
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
DEVICE_STR = DEVICE

if torch is None:
    logger.warning("PyTorch not available; speech pipeline will run on CPU.")
else:
    logger.info("[SpeechPipeline] Using device: %s", DEVICE_STR.upper())

TRANSCRIPT_CACHE_TIMEOUT = getattr(settings, "SPEECH_TRANSCRIPT_CACHE_TIMEOUT", 60 * 60 * 24 * 30)
ANALYSIS_CACHE_TIMEOUT = getattr(settings, "SPEECH_ANALYSIS_CACHE_TIMEOUT", 60 * 60 * 24 * 7)
WHISPER_MODEL_NAME = getattr(settings, "WHISPER_MODEL_NAME", "base")
WHISPER_LANGUAGE = getattr(settings, "WHISPER_LANGUAGE", "en")

_CACHE_PREFIX = "speech_pipeline"
_whisper_lock = threading.Lock()
_bert_lock = threading.Lock()
_hybrid_lock = threading.Lock()


def _stable_audio_hash(audio_path: str) -> str:
    """
    Generate a stable hash for an audio file based on path, size, and mtime.
    """
    try:
        stat = os.stat(audio_path)
        fingerprint = f"{audio_path}|{stat.st_size}|{stat.st_mtime_ns}"
    except (OSError, ValueError):
        fingerprint = audio_path
    return hashlib.sha256(fingerprint.encode("utf-8", "ignore")).hexdigest()


def _get_transcript_cache_key(audio_path: str) -> str:
    return f"{_CACHE_PREFIX}:transcript:{_stable_audio_hash(audio_path)}"


def _get_analysis_cache_key(audio_path: str = None, transcript: str = None, duration_seconds: float = None) -> str:
    parts = ["analysis"]
    if audio_path:
        parts.append(_stable_audio_hash(audio_path))
    if transcript:
        parts.append(hashlib.sha256(transcript.encode("utf-8", "ignore")).hexdigest())
    if duration_seconds is not None:
        parts.append(f"{round(duration_seconds, 2)}")
    key = ":".join(parts)
    return f"{_CACHE_PREFIX}:{key}"


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str, device: str) -> "torch.nn.Module | None":
    if torch is None:
        logger.warning("Cannot load Whisper model because PyTorch is unavailable.")
        return None
    try:
        import whisper
    except ImportError:
        logger.warning("openai-whisper is not installed; transcription will be disabled.")
        return None

    try:
        logger.info("Loading Whisper model '%s' on %s", model_name, device.upper())
        model = whisper.load_model(model_name, device=device)
        model.eval()
        return model
    except Exception as exc:
        logger.error("Failed to load Whisper model '%s': %s", model_name, exc)
        return None


@lru_cache(maxsize=1)
def _get_bert_tokenizer():
    from transformers import BertTokenizer

    logger.info("Loading BERT tokenizer (cached).")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


@lru_cache(maxsize=1)
def _get_bert_filler_model():
    from transformers import BertForSequenceClassification

    if torch is None:
        logger.warning("PyTorch not available; BERT filler model disabled.")
        return None

    model_dir = os.path.join("coach", "models", "bert_speech")
    if not os.path.exists(model_dir):
        logger.warning("BERT speech model directory not found at %s", model_dir)
        return None

    with _bert_lock:
        try:
            model = BertForSequenceClassification.from_pretrained(model_dir)
            model.to(DEVICE_STR)
            model.eval()
            return model
        except Exception as exc:
            logger.error("Failed to load BERT speech model: %s", exc)
            return None


@lru_cache(maxsize=1)
def _get_hybrid_model():
    if torch is None:
        logger.warning("PyTorch not available; hybrid speech model disabled.")
        return None

    model_dir = os.path.join("coach", "models", "hybrid_bert_wav2vec")
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        logger.warning("Hybrid model checkpoint not found at %s", model_path)
        return None

    with _hybrid_lock:
        try:
            from coach.train_bert import HybridSpeechModel

            model = HybridSpeechModel(num_labels=3)
            map_location = torch.device(DEVICE_STR)
            state_dict = torch.load(model_path, map_location=map_location)
            model.to(map_location)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as exc:
            logger.error("Failed to load hybrid speech model: %s", exc)
            return None


def send_verification_email(user):
    """
    Send email verification link to user.
    """
    signer = TimestampSigner()
    token = signer.sign(user.email)
    # Use reverse to get the full URL dynamically
    verification_link = settings.BASE_URL + reverse('coach:verify_email', args=[token])
    
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
    reset_link = settings.BASE_URL + reverse('coach:password_reset_confirm', args=[reset_token])
    
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


# Filler Word Detection Functions

def detect_filler_words(text):
    """
    Detect filler words in text using regex patterns.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary containing detected fillers and their positions
    """
    if not text:
        return {'fillers': [], 'count': 0, 'density': 0.0}
    
    # Common filler words and phrases
    filler_patterns = [
        r'\bum\b',
        r'\buh\b', 
        r'\blike\b',
        r'\byou know\b',
        r'\bso\b',
        r'\bwell\b',
        r'\bactually\b',
        r'\bbasically\b',
        r'\bkind of\b',
        r'\bsort of\b',
        r'\bI mean\b',
        r'\bright\b',
        r'\bokay\b',
        r'\byeah\b',
        r'\buhm\b',
        r'\ber\b',
        r'\berm\b'
    ]
    
    detected_fillers = []
    text_lower = text.lower()
    
    for pattern in filler_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            detected_fillers.append({
                'word': match.group(),
                'start': match.start(),
                'end': match.end(),
                'position': match.start()
            })
    
    # Calculate density (fillers per 100 words)
    word_count = len(text.split())
    density = (len(detected_fillers) / word_count * 100) if word_count > 0 else 0.0
    
    return {
        'fillers': detected_fillers,
        'count': len(detected_fillers),
        'density': round(density, 2)
    }


def get_filler_recommendations(filler_density):
    """
    Get recommendations based on filler word density.
    
    Args:
        filler_density (float): Filler words per 100 words
        
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    if filler_density > 10:
        recommendations.append("High filler density detected. Consider practicing pause techniques.")
        recommendations.append("Try the 'Pause Power-Up' drill to replace fillers with pauses.")
    elif filler_density > 5:
        recommendations.append("Moderate filler usage. Practice awareness techniques.")
        recommendations.append("Try the 'Filler Zap Game' to improve detection skills.")
    elif filler_density > 2:
        recommendations.append("Low filler usage. Good job! Continue practicing.")
        recommendations.append("Try the 'Silent Switcheroo' drill for advanced practice.")
    else:
        recommendations.append("Excellent! Very low filler usage.")
        recommendations.append("Try advanced drills to maintain your skills.")
    
    return recommendations


def analyze_speech_quality(text, duration_seconds=None):
    """
    Analyze overall speech quality including fillers, pacing, and clarity.
    
    Args:
        text (str): The transcribed text
        duration_seconds (float): Duration of speech in seconds
        
    Returns:
        dict: Analysis results
    """
    filler_analysis = detect_filler_words(text)
    
    # Calculate WPM if duration is provided
    word_count = len(text.split()) if text else 0
    wpm = (word_count / duration_seconds * 60) if duration_seconds and duration_seconds > 0 else 0
    
    # Determine pacing quality
    pacing_score = 100
    if wpm < 100:
        pacing_score -= (100 - wpm) * 0.5
    elif wpm > 180:
        pacing_score -= (wpm - 180) * 0.3
    
    # Determine clarity score based on fillers
    clarity_score = max(0, 100 - (filler_analysis['density'] * 5))
    
    # Overall confidence score
    confidence_score = (pacing_score + clarity_score) / 2
    
    return {
        'filler_analysis': filler_analysis,
        'wpm': round(wpm, 1),
        'pacing_score': round(max(0, pacing_score), 1),
        'clarity_score': round(max(0, clarity_score), 1),
        'confidence_score': round(max(0, confidence_score), 1),
        'recommendations': get_filler_recommendations(filler_analysis['density'])
    }


def get_drill_recommendations_by_analysis(analysis):
    """
    Get drill recommendations based on speech analysis.
    
    Args:
        analysis (dict): Speech analysis results
        
    Returns:
        list: List of recommended drill types
    """
    recommendations = []
    
    # Filler word recommendations
    if analysis['filler_analysis']['density'] > 5:
        recommendations.append('filler_words')
    
    # Pacing recommendations
    if analysis['wpm'] < 100 or analysis['wpm'] > 180:
        recommendations.append('pacing')
    
    # Pronunciation recommendations
    if analysis['clarity_score'] < 60:
        recommendations.append('pronunciation')
    
    return recommendations


# Sprint 3 NLP and ML Functions

def transcribe_audio(audio_file_path, *, language: str | None = None, force_refresh: bool = False):
    """
    Transcribe audio using OpenAI Whisper with GPU support and caching.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.
        language (str, optional): ISO language code override.
        force_refresh (bool): If True, bypasses the cache and reprocesses audio.

    Returns:
        dict: Transcription results with text, confidence, and metadata.
    """
    cache_key = _get_transcript_cache_key(audio_file_path)
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            cached_copy = dict(cached)
            cached_copy["from_cache"] = True
            return cached_copy

    model = _load_whisper_model(WHISPER_MODEL_NAME, DEVICE_STR)
    if model is None:
        return {
            "text": "",
            "language": language or WHISPER_LANGUAGE,
            "segments": [],
            "success": False,
            "error": "Whisper model unavailable",
        }

    transcription_language = language or WHISPER_LANGUAGE
    try:
        with _whisper_lock:
            result = model.transcribe(audio_file_path, language=transcription_language)

        payload = {
            "text": result.get("text", ""),
            "language": result.get("language", transcription_language),
            "segments": result.get("segments", []),
            "success": True,
            "error": None,
            "from_cache": False,
        }
        cache.set(cache_key, payload, TRANSCRIPT_CACHE_TIMEOUT)
        return payload
    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", audio_file_path, exc)
        return {
            "text": "",
            "language": transcription_language,
            "segments": [],
            "success": False,
            "error": str(exc),
        }


def analyze_speech(transcription_text, audio_duration=None):
    """
    Analyze speech using NLP models for comprehensive assessment.
    
    Args:
        transcription_text (str): The transcribed text to analyze
        audio_duration (float): Duration of audio in seconds (optional)
        
    Returns:
        dict: Comprehensive speech analysis results
    """
    try:
        from transformers import pipeline
        import numpy as np
        
        # Initialize sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer(transcription_text)
        
        # Combine with existing filler analysis
        filler_analysis = detect_filler_words(transcription_text)
        
        # Calculate additional metrics
        word_count = len(transcription_text.split())
        wpm = (word_count / audio_duration * 60) if audio_duration and audio_duration > 0 else 0
        
        # Calculate complexity metrics
        avg_word_length = np.mean([len(word) for word in transcription_text.split()]) if word_count > 0 else 0
        sentence_count = len([s for s in transcription_text.split('.') if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'sentiment': sentiment_result[0] if sentiment_result else {'label': 'NEUTRAL', 'score': 0.5},
            'filler_analysis': filler_analysis,
            'wpm': round(wpm, 1),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complexity_score': round((avg_word_length + avg_sentence_length) / 2, 2),
            'success': True,
            'error': None
        }
        
    except ImportError:
        # Fallback to basic analysis if transformers not available
        filler_analysis = detect_filler_words(transcription_text)
        word_count = len(transcription_text.split())
        wpm = (word_count / audio_duration * 60) if audio_duration and audio_duration > 0 else 0
        
        return {
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5},
            'filler_analysis': filler_analysis,
            'wpm': round(wpm, 1),
            'word_count': word_count,
            'sentence_count': len([s for s in transcription_text.split('.') if s.strip()]),
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'complexity_score': 0,
            'success': False,
            'error': 'Transformers not installed, using basic analysis'
        }
    except Exception as e:
        return {
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5},
            'filler_analysis': {'fillers': [], 'count': 0, 'density': 0.0},
            'wpm': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'complexity_score': 0,
            'success': False,
            'error': str(e)
        }


def train_cause_classifier():
    """
    Train a BERT-based classifier to identify causes of speech issues.
    
    Returns:
        dict: Training results and model information
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        from datasets import Dataset
        import torch
        from sklearn.model_selection import train_test_split
        
        # Placeholder for training data - in real implementation, this would come from database
        training_data = [
            {"text": "I um, like, you know, get nervous when speaking", "label": "anxiety"},
            {"text": "So basically, the thing is that, uh, I'm not sure", "label": "lack_of_confidence"},
            {"text": "Well, actually, I think that, like, it's really hard", "label": "poor_skills"},
            {"text": "I'm stressed about this presentation tomorrow", "label": "stress"},
        ]
        
        # Convert to dataset format
        dataset = Dataset.from_list(training_data)
        
        # Split data
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Load pre-trained model and tokenizer
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=4,  # anxiety, stress, lack_of_confidence, poor_skills
            id2label={0: "anxiety", 1: "stress", 2: "lack_of_confidence", 3: "poor_skills"},
            label2id={"anxiety": 0, "stress": 1, "lack_of_confidence": 2, "poor_skills": 3}
        )
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./cause_classifier",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        return {
            'success': True,
            'model_path': './cause_classifier',
            'training_samples': len(training_data),
            'accuracy': 0.85,  # Placeholder - would be calculated from actual training
            'error': None
        }
        
    except ImportError as e:
        return {
            'success': False,
            'model_path': None,
            'training_samples': 0,
            'accuracy': 0,
            'error': f'Required packages not installed: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'model_path': None,
            'training_samples': 0,
            'accuracy': 0,
            'error': str(e)
        }


def train_recommender():
    """
    Train a recommendation system for personalized drill suggestions.
    
    Returns:
        dict: Training results and model information
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import KMeans
        import numpy as np
        import pickle
        
        # Placeholder training data - in real implementation, this would come from user sessions
        user_sessions = [
            {"user_id": 1, "filler_density": 8.5, "wpm": 120, "clarity_score": 65, "preferred_drills": ["filler_zap_game", "pause_powerup"]},
            {"user_id": 2, "filler_density": 3.2, "wpm": 95, "clarity_score": 80, "preferred_drills": ["metronome_rhythm", "timer_tag_team"]},
            {"user_id": 3, "filler_density": 12.1, "wpm": 180, "clarity_score": 45, "preferred_drills": ["silent_switcheroo", "echo_elimination"]},
        ]
        
        # Extract features
        features = []
        drill_preferences = []
        
        for session in user_sessions:
            features.append([session["filler_density"], session["wpm"], session["clarity_score"]])
            drill_preferences.append(session["preferred_drills"])
        
        features = np.array(features)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Cluster users based on speech patterns
        kmeans = KMeans(n_clusters=3, random_state=42)
        user_clusters = kmeans.fit_predict(features_normalized)
        
        # Create drill recommendation matrix
        all_drills = ["filler_zap_game", "pause_powerup", "metronome_rhythm", "timer_tag_team", 
                      "silent_switcheroo", "echo_elimination", "poem_echo", "vowel_vortex"]
        
        # Build recommendation model
        recommendation_model = {
            'scaler': scaler,
            'kmeans': kmeans,
            'drill_clusters': {},
            'drill_similarity': {}
        }
        
        # Assign drills to clusters based on user preferences
        for cluster_id in range(3):
            cluster_users = [i for i, c in enumerate(user_clusters) if c == cluster_id]
            cluster_drills = []
            for user_idx in cluster_users:
                cluster_drills.extend(drill_preferences[user_idx])
            recommendation_model['drill_clusters'][cluster_id] = list(set(cluster_drills))
        
        # Calculate drill similarity using TF-IDF
        drill_descriptions = {
            "filler_zap_game": "eliminate filler words game interactive",
            "pause_powerup": "replace fillers with pauses timer",
            "metronome_rhythm": "pacing rhythm timing metronome",
            "timer_tag_team": "alternating speech pause timing",
            "silent_switcheroo": "rewrite sentences remove fillers",
            "echo_elimination": "echo playback filler removal",
            "poem_echo": "pronunciation poem reading",
            "vowel_vortex": "vowel pronunciation practice"
        }
        
        vectorizer = TfidfVectorizer()
        drill_vectors = vectorizer.fit_transform(drill_descriptions.values())
        drill_similarity = cosine_similarity(drill_vectors)
        
        recommendation_model['drill_similarity'] = {
            drill: similarity.tolist() 
            for drill, similarity in zip(drill_descriptions.keys(), drill_similarity)
        }
        
        # Save model
        with open('media/recommendation_model.pkl', 'wb') as f:
            pickle.dump(recommendation_model, f)
        
        return {
            'success': True,
            'model_path': 'media/recommendation_model.pkl',
            'training_samples': len(user_sessions),
            'clusters': len(set(user_clusters)),
            'drills_available': len(all_drills),
            'error': None
        }
        
    except ImportError as e:
        return {
            'success': False,
            'model_path': None,
            'training_samples': 0,
            'clusters': 0,
            'drills_available': 0,
            'error': f'Required packages not installed: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'model_path': None,
            'training_samples': 0,
            'drills_available': 0,
            'error': str(e)
        }


# BERT-based Speech Analysis Functions

def get_bert_scores(transcript: str, audio_path: str = None) -> dict:
    """
    Optimized BERT scoring: Use hybrid model when audio is available, pure BERT otherwise.
    
    Strategy:
    - If audio_path exists: Use Hybrid BERT+Wav2Vec2 for all 3 scores (filler, clarity, pacing)
    - If audio_path missing: Use Pure BERT for all 3 scores
    - Fallback: Pure BERT if hybrid fails, rule-based if both fail
    
    This eliminates redundant computation and uses models as they were trained.
    
    Args:
        transcript (str): The text to analyze
        audio_path (str, optional): Path to audio file for Wav2Vec2 features
        
    Returns:
        dict: Scores for filler, clarity, and pacing (0.0-1.0, higher = worse)
    """
    if torch is None:
        logger.warning("PyTorch unavailable, falling back to heuristic BERT scores.")
        return get_fallback_bert_scores(transcript)

    try:
        tokenizer = _get_bert_tokenizer()
        inputs = tokenizer(
            transcript,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: tensor.to(DEVICE_STR) for key, tensor in inputs.items()}

        filler_score = None
        clarity_score = None
        pacing_score = None

        # Strategy: If audio is available, try hybrid model first (uses models as trained)
        if audio_path and os.path.exists(audio_path):
            hybrid_model = _get_hybrid_model()
            if hybrid_model is not None:
                try:
                    from coach.audio_features import get_wav2vec_features

                    wav2vec_features = get_wav2vec_features(audio_path)
                    wav2vec_tensor = torch.tensor(
                        wav2vec_features, dtype=torch.float32, device=DEVICE_STR
                    ).unsqueeze(0)

                    with torch.no_grad():
                        logits = hybrid_model(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            wav2vec_tensor,
                        )
                        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
                        filler_score = float(probs[0])
                        clarity_score = float(probs[1])
                        pacing_score = float(probs[2])
                        logger.debug(
                            "Hybrid model scores -> filler: %.3f, clarity: %.3f, pacing: %.3f",
                            filler_score,
                            clarity_score,
                            pacing_score,
                        )
                except Exception as exc:
                    logger.warning("Hybrid model inference failed: %s, falling back to pure BERT", exc)

        # Fallback: Use pure BERT if hybrid failed or audio not available
        if filler_score is None or clarity_score is None or pacing_score is None:
            filler_model = _get_bert_filler_model()
            if filler_model is not None:
                try:
                    with torch.no_grad():
                        outputs = filler_model(**inputs)
                        probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
                        filler_score = float(probs[0])
                        clarity_score = float(probs[1])
                        pacing_score = float(probs[2])
                        logger.debug(
                            "Pure BERT scores -> filler: %.3f, clarity: %.3f, pacing: %.3f",
                            filler_score,
                            clarity_score,
                            pacing_score,
                        )
                except Exception as exc:
                    logger.warning("Pure BERT inference failed: %s", exc)

        # Final fallback: Rule-based scores
        if filler_score is None or clarity_score is None or pacing_score is None:
            logger.warning("All models failed, using rule-based scores")
            fallback = get_fallback_bert_scores(transcript)
            return {
                "filler": filler_score if filler_score is not None else fallback["filler"],
                "clarity": clarity_score if clarity_score is not None else fallback["clarity"],
                "pacing": pacing_score if pacing_score is not None else fallback["pacing"],
            }

        return {
            "filler": filler_score,
            "clarity": clarity_score,
            "pacing": pacing_score,
        }

    except Exception as exc:
        logger.error("Error in BERT scoring: %s", exc)
        return get_fallback_bert_scores(transcript)


def get_fallback_bert_scores(transcript: str) -> dict:
    """
    Fallback rule-based scores if BERT model not available.
    
    Args:
        transcript (str): The text to analyze
        
    Returns:
        dict: Scores for filler, clarity, and pacing
    """
    # Filler score (based on filler word density)
    filler_analysis = detect_filler_words(transcript)
    filler_score = min(1.0, filler_analysis['density'] / 10.0)
    
    # Clarity score (inverse of filler score for simplicity)
    clarity_score = max(0.0, 1.0 - filler_score)
    
    # Pacing score (based on sentence structure)
    sentences = transcript.split('.')
    avg_sentence_length = len(transcript.split()) / max(len(sentences), 1)
    pacing_score = 0.5  # Neutral by default
    
    return {
        'filler': filler_score,
        'clarity': clarity_score,
        'pacing': pacing_score
    }


def predict_cause(audio_path: str, transcript: str, duration: float = None) -> dict:
    """
    Predict root cause using XGBoost classifier.
    
    Args:
        audio_path: Path to audio file
        transcript: Transcribed text
        duration: Audio duration in seconds (optional)
        
    Returns:
        dict: {
            'cause': 'anxiety' | 'stress' | 'lack_of_confidence' | 'poor_skills',
            'confidence': float,
            'features': dict
        }
    """
    try:
        import pickle
        import pandas as pd
        import librosa
        import numpy as np
        import os
        
        model_path = os.path.join('coach', 'models', 'cause_model.pkl')
        scaler_path = os.path.join('coach', 'models', 'scaler.pkl')
        encoder_path = os.path.join('coach', 'models', 'label_encoder.pkl')
        
        # Check if models exist
        if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
            logger.warning("Cause model not found, using rule-based prediction")
            return predict_cause_fallback(transcript, duration)
        
        # Load models
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Get BERT scores
        bert_scores = get_bert_scores(transcript)
        
        # Load audio if available
        if audio_path and os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=16000, duration=30)
                duration = len(y) / sr
                
                # Compute audio features
                intervals = librosa.effects.split(y, top_db=20)
                silence_dur = sum((end - start) / sr for start, end in intervals)
                pause_ratio = silence_dur / duration if duration > 0 else 0
                
                # Speech rate variance (simplified)
                speech_rate_var = 0.0
            except Exception as e:
                logger.warning(f"Could not load audio: {e}, using default values")
                pause_ratio = 0.0
                speech_rate_var = 0.0
        else:
            # Fallback if no audio
            pause_ratio = 0.0
            speech_rate_var = 0.0
            if not duration:
                duration = len(transcript.split()) / 3  # Estimate ~180 WPM
        
        # Calculate WPM
        words = len(transcript.split())
        wpm = words / (duration / 60) if duration > 0 else 0
        
        # Create feature vector
        features = {
            'bert_filler': bert_scores['filler'],
            'bert_clarity': bert_scores['clarity'],
            'bert_pacing': bert_scores['pacing'],
            'wpm': wpm,
            'pause_ratio': pause_ratio,
            'speech_rate_var': speech_rate_var
        }
        
        # Scale features
        X = pd.DataFrame([features])
        X_scaled = scaler.transform(X)
        
        # Predict
        y_pred_numeric = model.predict(X_scaled)[0]
        y_proba = model.predict_proba(X_scaled)[0]
        
        # Decode to label
        cause = label_encoder.inverse_transform([y_pred_numeric])[0]
        confidence = float(max(y_proba))
        
        return {
            'cause': cause,
            'confidence': confidence,
            'features': features,
            'all_probabilities': dict(zip(label_encoder.classes_, y_proba))
        }
        
    except Exception as e:
        logger.error(f"Error in cause prediction: {e}")
        return predict_cause_fallback(transcript, duration)


def predict_cause_fallback(transcript: str, duration: float = None) -> dict:
    """
    Fallback rule-based cause prediction.
    """
    # Calculate basic metrics
    words = len(transcript.split())
    if not duration:
        duration = words / 3  # Estimate ~180 WPM
    wpm = words / (duration / 60) if duration > 0 else 0
    
    # Filler density
    fillers = ['um', 'uh', 'like', 'you know', 'so', 'well', 'i mean', 'literally', 'actually', 'basically']
    filler_count = sum(transcript.lower().count(f) for f in fillers)
    filler_density = filler_count / duration if duration > 0 else 0
    
    # Rule-based classification
    if filler_density > 0.05 and wpm > 160:
        cause = 'anxiety'
    elif wpm < 110 and filler_count > 10:
        cause = 'stress'
    elif filler_density > 0.03:
        cause = 'lack_of_confidence'
    else:
        cause = 'poor_skills'
    
    return {
        'cause': cause,
        'confidence': 0.6,  # Lower confidence for fallback
        'features': {'wpm': wpm, 'filler_density': filler_density},
        'all_probabilities': {}
    }


def analyze_speech_pipeline(
    audio_path: str | None,
    transcript: str | None = None,
    duration_seconds: float | None = None,
    *,
    force_refresh: bool = False,
) -> dict:
    """
    End-to-end speech analysis pipeline with GPU acceleration, caching, and fallbacks.

    Args:
        audio_path: Path to the audio file (optional if transcript provided).
        transcript: Pre-computed transcript (optional).
        duration_seconds: Length of the audio in seconds.
        force_refresh: Bypass caches when True.

    Returns:
        dict: Aggregated analysis payload.
    """
    cache_key = _get_analysis_cache_key(audio_path, transcript, duration_seconds)
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            cached_copy = dict(cached)
            cached_copy["from_cache"] = True
            return cached_copy

    try:
        transcription_result = None
        resolved_transcript = transcript or ""

        if not resolved_transcript and audio_path:
            transcription_result = transcribe_audio(audio_path, force_refresh=force_refresh)
            if not transcription_result.get("success"):
                return {
                    "success": False,
                    "error": transcription_result.get("error", "Transcription failed"),
                    "transcription": transcription_result,
                    "transcript": transcription_result.get("text", ""),
                    "from_cache": False,
                }
            resolved_transcript = transcription_result.get("text", "")

        detected_language = None
        if transcription_result:
            detected_language = transcription_result.get("language")
        elif transcript:
            detected_language = None
        else:
            detected_language = WHISPER_LANGUAGE

        if detected_language and detected_language.lower() not in {"en", "en-us", "en-gb"}:
            logger.warning(
                "Detected non-English language (%s) for audio %s", detected_language, audio_path
            )
            return {
                "success": False,
                "error": (
                    "Detected language mismatch: we only support English audio at the moment. "
                    f"Please provide English speech instead of '{detected_language}'."
                ),
                "transcription": transcription_result,
                "transcript": resolved_transcript,
                "from_cache": False,
                "language": detected_language,
            }

        quality = analyze_speech_quality(resolved_transcript, duration_seconds)
        bert_scores = get_bert_scores(resolved_transcript, audio_path=audio_path)
        cause_prediction = predict_cause(
            audio_path=audio_path or "",
            transcript=resolved_transcript,
            duration=duration_seconds,
        )

        result = {
            "success": True,
            "transcript": resolved_transcript,
            "quality": quality,
            "bert_scores": bert_scores,
            "cause_prediction": cause_prediction,
            "transcription": transcription_result,
            "from_cache": False,
            "language": detected_language or WHISPER_LANGUAGE,
        }
        cache.set(cache_key, result, ANALYSIS_CACHE_TIMEOUT)
        return result

    except Exception as exc:
        logger.error("End-to-end speech analysis failed: %s", exc)
        return {
            "success": False,
            "error": str(exc),
            "transcript": transcript or "",
            "from_cache": False,
        }


class DrillRecommender:
    """
    Recommends optimal drill based on analysis scores and user history.
    
    Logic:
    - Filler score > 0.6 → Filler Words drills
    - Pacing score > 0.6 → Pacing drills
    - Clarity score > 0.6 → Pronunciation drills
    - ML-based personalized recommendation based on history
    """
    
    def __init__(self):
        from .models import Drill
        self.Drill = Drill
        self.logger = logging.getLogger(__name__)
    
    def pick_best(self, scores: dict, history: list = None) -> 'Drill':
        """
        Pick the best drill based on current analysis and user history.
        
        Args:
            scores (dict): Dictionary with 'filler', 'clarity', 'pacing' scores (0-1)
                          Higher score = worse performance (more fillers, worse clarity, worse pacing)
            history (list): List of recent Feedback objects for context
        
        Returns:
            Drill: Recommended drill object
        """
        try:
            # Priority-based recommendation
            filler_score = scores.get('filler', 0.0)
            clarity_score = scores.get('clarity', 0.0)
            pacing_score = scores.get('pacing', 0.0)
            
            self.logger.info(f"Drill recommendation scores - Filler: {filler_score:.3f}, Clarity: {clarity_score:.3f}, Pacing: {pacing_score:.3f}")
            
            # Find worst-performing area (highest score = worst)
            worst_area = max([
                ('filler', filler_score),
                ('clarity', clarity_score),
                ('pacing', pacing_score)
            ], key=lambda x: x[1])
            
            area, score = worst_area
            
            # Also consider actual performance metrics if available (from session/feedback)
            # This helps cross-validate model scores with real metrics
            actual_metrics = scores.get('_actual_metrics', {})
            wpm = actual_metrics.get('wpm', None)
            filler_count = actual_metrics.get('filler_count', None)
            filler_density = actual_metrics.get('filler_density', None)  # Percentage
            
            # Adjust scores based on actual metrics to catch model errors
            adjusted_scores = {
                'filler': filler_score,
                'clarity': clarity_score,
                'pacing': pacing_score
            }
            
            # Cross-validate filler score with actual filler metrics
            if filler_count is not None or filler_density is not None:
                # If actual fillers are low (< 5/min or < 3% density), reduce filler score
                if filler_density is not None and filler_density < 3.0:
                    adjusted_scores['filler'] = max(0.0, filler_score - 0.4)  # Reduce score significantly
                    self.logger.info(f"Filler score adjusted: {filler_score:.3f} -> {adjusted_scores['filler']:.3f} (actual density: {filler_density:.2f}%)")
                elif filler_count is not None and wpm is not None and wpm > 0:
                    fillers_per_min = (filler_count / (wpm / 60.0)) if wpm > 0 else 0
                    if fillers_per_min < 5.0:  # Less than 5 fillers per minute is excellent
                        adjusted_scores['filler'] = max(0.0, filler_score - 0.4)
                        self.logger.info(f"Filler score adjusted: {filler_score:.3f} -> {adjusted_scores['filler']:.3f} (actual: {fillers_per_min:.1f}/min)")
            
            # Cross-validate pacing score with actual WPM
            if wpm is not None:
                # Optimal WPM is 120-180
                if wpm < 120:  # Too slow
                    adjusted_scores['pacing'] = min(1.0, pacing_score + 0.3)  # Increase score (worse)
                    self.logger.info(f"Pacing score adjusted: {pacing_score:.3f} -> {adjusted_scores['pacing']:.3f} (WPM: {wpm:.0f} - too slow)")
                elif wpm > 180:  # Too fast
                    adjusted_scores['pacing'] = min(1.0, pacing_score + 0.2)  # Increase score (worse)
                    self.logger.info(f"Pacing score adjusted: {pacing_score:.3f} -> {adjusted_scores['pacing']:.3f} (WPM: {wpm:.0f} - too fast)")
                elif 120 <= wpm <= 180:  # Optimal range
                    adjusted_scores['pacing'] = max(0.0, pacing_score - 0.2)  # Decrease score (better)
                    self.logger.info(f"Pacing score adjusted: {pacing_score:.3f} -> {adjusted_scores['pacing']:.3f} (WPM: {wpm:.0f} - optimal)")
            
            # Use adjusted scores for recommendation
            filler_score = adjusted_scores['filler']
            clarity_score = adjusted_scores['clarity']
            pacing_score = adjusted_scores['pacing']
            
            # Find worst-performing area using adjusted scores
            scores_list = [
                ('filler', filler_score),
                ('clarity', clarity_score),
                ('pacing', pacing_score)
            ]
            scores_list.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
            
            # If top score is significantly higher than others (>0.15 difference), use it
            if len(scores_list) >= 2 and scores_list[0][1] - scores_list[1][1] > 0.15:
                area, score = scores_list[0]
                self.logger.info(f"Clear worst area: {area} (adjusted score: {score:.3f})")
            # If all scores are very low (<0.3), all performance is good - use history if available
            elif score < 0.3:
                if history and len(history) >= 3:
                    # Analyze history to find most common issue
                    avg_filler = sum(f.filler_score if f.filler_score else 0 for f in history[-5:]) / len(history[-5:])
                    avg_clarity = sum(f.clarity_score if f.clarity_score else 0 for f in history[-5:]) / len(history[-5:])
                    avg_pacing = sum(f.pacing_score if f.pacing_score else 0 for f in history[-5:]) / len(history[-5:])
                    
                    worst_area = max([
                        ('filler', avg_filler),
                        ('clarity', avg_clarity),
                        ('pacing', avg_pacing)
                    ], key=lambda x: x[1])
                    area, score = worst_area
                    self.logger.info(f"Using history - worst area: {area} (avg score: {score:.3f})")
                else:
                    # No history, default to pacing if all scores are low
                    # (pacing is often the most common improvement area)
                    area = 'pacing'
                    self.logger.info(f"No history available, defaulting to pacing")
            else:
                # Use the worst area from current scores
                area, score = worst_area
                self.logger.info(f"Using current scores - worst area: {area} (score: {score:.3f})")
            
            # Map area to skill type
            skill_type_map = {
                'filler': 'filler_words',
                'clarity': 'pronunciation',
                'pacing': 'pacing'
            }
            skill_type = skill_type_map.get(area, 'pronunciation')
            
            # Get random drill from this category
            drills = self.Drill.objects.filter(
                skill_type=skill_type,
                is_active=True
            )
            
            if drills.exists():
                drill = drills.order_by('?').first()  # Random drill
                self.logger.info(f"Recommended {skill_type} drill: {drill.name} (score: {score:.3f})")
                return drill
            else:
                # Fallback to any active drill
                fallback = self.Drill.objects.filter(is_active=True).first()
                self.logger.warning(f"No drills found for {skill_type}, using fallback")
                return fallback
            
        except Exception as e:
            self.logger.error(f"Error in drill recommendation: {e}")
            # Return first available drill as fallback
            return self.Drill.objects.filter(is_active=True).first()
    
    def ml_predict(self, scores: dict, history: list) -> 'Drill':
        """
        ML-based personalized recommendation (future enhancement).
        For now, uses pick_best logic.
        """
        return self.pick_best(scores, history)

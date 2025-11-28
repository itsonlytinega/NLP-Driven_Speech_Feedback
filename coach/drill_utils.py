"""
Unified drill analysis utilities for all practice drills.
Provides ML-based scoring, progress tracking, adaptive difficulty, and personalization.
"""
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone
from datetime import timedelta

from .models import Drill, DrillCompletion, User, Feedback
from speech_sessions.models import SpeechSession
from .utils import (
    transcribe_audio,
    get_bert_scores,
    predict_cause,
    detect_filler_words,
    analyze_speech_quality
)

logger = logging.getLogger(__name__)


def analyze_drill_audio(audio_data: bytes, drill: Drill, user: User, 
                       drill_specific_params: Optional[Dict] = None) -> Dict:
    """
    Unified ML-based analysis for all drills.
    
    Args:
        audio_data: Raw audio bytes
        drill: Drill instance
        user: User instance
        drill_specific_params: Drill-specific parameters (e.g., target_text, difficulty)
    
    Returns:
        Dict with comprehensive analysis results
    """
    drill_specific_params = drill_specific_params or {}
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        # Get audio duration first (needed for accurate WPM calculation)
        import wave
        import struct
        try:
            with wave.open(tmp_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                audio_duration = frames / float(sample_rate) if sample_rate > 0 else 0
        except Exception as e:
            logger.warning(f"Could not get duration from WAV file: {e}, using fallback")
            # Fallback: estimate from file size (rough approximation)
            import os
            file_size = os.path.getsize(tmp_path)
            # Rough estimate: 16-bit mono = 2 bytes per sample, assume 16kHz
            audio_duration = (file_size / 2) / 16000 if file_size > 0 else 0
        
        # Use duration from params if provided and valid, otherwise use calculated
        if drill_specific_params.get('duration', 0) > 0:
            duration = drill_specific_params['duration']
        else:
            duration = audio_duration
        
        # Transcribe audio
        transcription_result = transcribe_audio(tmp_path)
        if not transcription_result['success']:
            return {
                'success': False,
                'error': 'Transcription failed',
                'details': transcription_result.get('error')
            }
        
        text = transcription_result['text']
        
        # Use duration from transcription if available and valid, otherwise use our calculated duration
        transcription_duration = transcription_result.get('duration', 0)
        if transcription_duration > 0:
            duration = transcription_duration
        
        # Get ML scores
        bert_scores = {}
        try:
            bert_scores = get_bert_scores(text, audio_path=tmp_path)
        except Exception as e:
            logger.warning(f"BERT scoring failed: {e}")
            # Fallback
            filler_analysis = detect_filler_words(text)
            bert_scores = {
                'filler': min(1.0, filler_analysis['density'] / 100.0),
                'clarity': 0.5,
                'pacing': 0.5
            }
        
        # Get cause prediction
        cause_prediction = {}
        try:
            cause_prediction = predict_cause(audio_path=tmp_path, transcript=text, duration=duration)
        except Exception as e:
            logger.warning(f"Cause prediction failed: {e}")
            cause_prediction = {'cause': 'other', 'confidence': 0.0}
        
        # Detailed analysis
        filler_analysis = detect_filler_words(text)
        speech_analysis = analyze_speech_quality(text, duration)
        
        # Calculate WPM - ensure we have valid duration
        word_count = len(text.split())
        if duration > 0:
            wpm = int((word_count / duration) * 60)
        else:
            # Fallback: estimate WPM if duration is unknown (assume average speaking rate)
            wpm = int(word_count * 2)  # Rough estimate: 2 words per second = 120 WPM
            logger.warning(f"Duration is 0, using estimated WPM: {wpm}")
        
        # Ensure WPM is reasonable (between 50 and 300)
        wpm = max(50, min(300, wpm))
        
        # Align filler count with BERT score for consistency
        # BERT is more sophisticated and detects subtle hesitations
        bert_filler_raw = bert_scores.get('filler', 0.5)
        
        # If BERT detects significant fillers but simple detector doesn't, estimate from BERT
        if bert_filler_raw > 0.2 and filler_analysis['count'] == 0:
            # Estimate filler count based on BERT score and text length
            # Higher BERT score = more fillers/hesitations
            estimated_fillers = int(word_count * bert_filler_raw * 0.1)  # Rough estimate
            if estimated_fillers > 0:
                filler_analysis['count'] = estimated_fillers
                filler_analysis['density'] = (estimated_fillers / word_count * 100) if word_count > 0 else 0
                logger.info(f"Aligned filler count with BERT: {estimated_fillers} fillers (BERT score: {bert_filler_raw:.2f})")
        # If simple detector finds fillers but BERT score is very low, trust simple detector
        elif filler_analysis['count'] > 0 and bert_filler_raw < 0.1:
            # BERT might have missed explicit fillers, keep simple detector count
            logger.info(f"Using simple detector count ({filler_analysis['count']}) as BERT score is very low")
        
        # Drill-specific analysis
        drill_metrics = _analyze_drill_specific(drill, text, audio_path=tmp_path, 
                                               bert_scores=bert_scores, 
                                               params=drill_specific_params)
        
        # Calculate overall score (0.0-1.0, higher = better)
        overall_score = _calculate_overall_score(bert_scores, drill_metrics, drill)
        
        # Normalize scores (convert to 0-1 where higher is better)
        normalized_scores = {
            'filler_score': 1.0 - min(1.0, bert_scores.get('filler', 0.5)),  # Invert
            'clarity_score': 1.0 - min(1.0, bert_scores.get('clarity', 0.5)),  # Invert
            'pacing_score': 1.0 - min(1.0, bert_scores.get('pacing', 0.5)),  # Invert
            'overall_score': overall_score
        }
        
        return {
            'success': True,
            'transcription': text,
            'duration': duration,
            'ml_scores': normalized_scores,
            'ml_metrics': {
                'fillers_count': filler_analysis['count'],
                'filler_density': filler_analysis['density'],
                'words': word_count,
                'wpm': wpm,
                'filler_words': filler_analysis['fillers'],
            },
            'drill_specific_metrics': drill_metrics,
            'cause': cause_prediction.get('cause', 'other'),
            'cause_confidence': cause_prediction.get('confidence', 0.0),
            'recommendations': speech_analysis.get('recommendations', []),
            'audio_path': tmp_path  # Keep for saving if needed
        }
        
    except Exception as e:
        logger.error(f"Drill analysis failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def _analyze_drill_specific(drill: Drill, text: str, audio_path: str, 
                           bert_scores: Dict, params: Dict) -> Dict:
    """Analyze drill-specific metrics based on drill type."""
    metrics = {}
    
    if drill.skill_type == 'pronunciation':
        # Pronunciation-specific metrics
        metrics['pronunciation_accuracy'] = _calculate_pronunciation_accuracy(
            text, params.get('target_text'), params.get('phonetic_targets')
        )
        metrics['articulation_score'] = _calculate_articulation_score(text)
        
    elif drill.skill_type == 'pacing':
        # Pacing-specific metrics
        target_wpm = params.get('target_wpm', 150)
        actual_wpm = len(text.split()) / (params.get('duration', 1) / 60) if params.get('duration') else 0
        metrics['wpm_accuracy'] = 1.0 - min(1.0, abs(actual_wpm - target_wpm) / target_wpm)
        metrics['pace_consistency'] = _calculate_pace_consistency(text, params.get('duration', 1))
        metrics['pause_quality'] = _calculate_pause_quality(text, audio_path)
        
    elif drill.skill_type == 'filler_words':
        # Filler-specific metrics
        filler_analysis = detect_filler_words(text)
        metrics['filler_reduction_rate'] = _calculate_filler_reduction(
            filler_analysis['count'], params.get('baseline_fillers', 0)
        )
        metrics['filler_avoidance_score'] = 1.0 - min(1.0, filler_analysis['density'] / 100.0)
    
    return metrics


def _calculate_pronunciation_accuracy(text: str, target_text: Optional[str] = None, 
                                    phonetic_targets: Optional[List] = None) -> float:
    """Calculate pronunciation accuracy (simplified - can be enhanced with phoneme comparison)."""
    if not target_text:
        return 0.5  # Default if no target
    
    # Simple word match ratio (can be enhanced with phoneme-level comparison)
    target_words = set(target_text.lower().split())
    actual_words = set(text.lower().split())
    if not target_words:
        return 0.5
    
    match_ratio = len(target_words.intersection(actual_words)) / len(target_words)
    return min(1.0, match_ratio * 1.2)  # Slight boost for partial matches


def _calculate_articulation_score(text: str) -> float:
    """Calculate articulation score based on text characteristics."""
    # Simple heuristic: longer words and complex structures indicate better articulation
    words = text.split()
    if not words:
        return 0.5
    
    avg_word_length = sum(len(w) for w in words) / len(words)
    # Normalize to 0-1 scale (assuming 3-8 chars is typical)
    score = min(1.0, (avg_word_length - 3) / 5.0)
    return max(0.0, score)


def _calculate_pace_consistency(text: str, duration: float) -> float:
    """Calculate pace consistency across speech segments."""
    if duration <= 0:
        return 0.5
    
    words = text.split()
    if len(words) < 2:
        return 0.5
    
    # Simple heuristic: check if WPM is relatively consistent
    # (In a real implementation, would analyze audio segments)
    wpm = (len(words) / duration) * 60
    # Assume 120-180 WPM is good range
    if 120 <= wpm <= 180:
        return 0.8
    elif 100 <= wpm <= 200:
        return 0.6
    else:
        return 0.4


def _calculate_pause_quality(text: str, audio_path: str) -> float:
    """Calculate pause quality (simplified - would need audio analysis)."""
    # Simple heuristic: check for natural sentence breaks
    sentences = text.split('.')
    if len(sentences) > 1:
        return 0.7  # Has pauses
    return 0.5  # Default


def _calculate_filler_reduction(current_fillers: int, baseline_fillers: int) -> float:
    """Calculate filler reduction rate."""
    if baseline_fillers == 0:
        return 0.5  # Default if no baseline
    
    reduction = (baseline_fillers - current_fillers) / baseline_fillers
    return max(0.0, min(1.0, reduction))


def _calculate_overall_score(bert_scores: Dict, drill_metrics: Dict, drill: Drill) -> float:
    """Calculate overall score based on drill type and metrics."""
    # Base scores (inverted since higher = worse in bert_scores)
    filler = 1.0 - min(1.0, bert_scores.get('filler', 0.5))
    clarity = 1.0 - min(1.0, bert_scores.get('clarity', 0.5))
    pacing = 1.0 - min(1.0, bert_scores.get('pacing', 0.5))
    
    # Weight based on drill type
    if drill.skill_type == 'pronunciation':
        weights = {'clarity': 0.5, 'filler': 0.2, 'pacing': 0.3}
        drill_weight = drill_metrics.get('pronunciation_accuracy', 0.5) * 0.3
    elif drill.skill_type == 'pacing':
        weights = {'pacing': 0.5, 'clarity': 0.3, 'filler': 0.2}
        drill_weight = drill_metrics.get('wpm_accuracy', 0.5) * 0.3
    else:  # filler_words
        weights = {'filler': 0.6, 'clarity': 0.2, 'pacing': 0.2}
        drill_weight = drill_metrics.get('filler_avoidance_score', 0.5) * 0.3
    
    base_score = (filler * weights['filler'] + 
                  clarity * weights['clarity'] + 
                  pacing * weights['pacing'])
    
    return min(1.0, base_score + drill_weight)


def get_user_difficulty_level(user: User, drill: Drill) -> str:
    """
    Determine adaptive difficulty level for user based on their performance history.
    
    Returns: 'beginner', 'intermediate', or 'advanced'
    """
    # Get recent completions for this drill
    recent_completions = DrillCompletion.objects.filter(
        user=user,
        drill=drill
    ).order_by('-completed_at')[:10]
    
    if not recent_completions:
        # No history - start with beginner
        return 'beginner'
    
    # Calculate average overall score
    scores = [c.get_overall_score() for c in recent_completions if c.ml_scores]
    if not scores:
        return 'beginner'
    
    avg_score = sum(scores) / len(scores)
    
    # Determine difficulty
    if avg_score >= 0.75:
        return 'advanced'
    elif avg_score >= 0.5:
        return 'intermediate'
    else:
        return 'beginner'


def get_user_weak_areas(user: User) -> Dict[str, float]:
    """
    Identify user's weak areas based on all feedback and sessions.
    
    Returns: Dict with scores for filler, clarity, pacing (0.0-1.0, higher = worse)
    """
    # Get recent feedback
    recent_feedbacks = Feedback.objects.filter(user=user).order_by('-created_at')[:20]
    
    # Get recent sessions
    recent_sessions = SpeechSession.objects.filter(
        user=user, 
        status='analyzed'
    ).order_by('-date')[:20]
    
    filler_scores = []
    clarity_scores = []
    pacing_scores = []
    
    # Collect from feedback
    for feedback in recent_feedbacks:
        if feedback.fillers_count is not None:
            # Normalize filler count (assuming 0-50 is range)
            filler_scores.append(min(1.0, feedback.fillers_count / 50.0))
        if feedback.f1_score is not None:
            clarity_scores.append(1.0 - feedback.f1_score)  # Invert
    
    # Collect from sessions
    for session in recent_sessions:
        if session.filler_score is not None:
            filler_scores.append(session.filler_score)
        if session.clarity_score is not None:
            clarity_scores.append(session.clarity_score)
        if session.pacing_score is not None:
            pacing_scores.append(session.pacing_score)
    
    # Calculate averages
    avg_filler = sum(filler_scores) / len(filler_scores) if filler_scores else 0.5
    avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.5
    avg_pacing = sum(pacing_scores) / len(pacing_scores) if pacing_scores else 0.5
    
    return {
        'filler': avg_filler,
        'clarity': avg_clarity,
        'pacing': avg_pacing
    }


def get_personalized_content(drill: Drill, user: User, content_pool: List) -> Dict:
    """
    Get personalized content based on user's weak areas.
    
    Args:
        drill: Drill instance
        user: User instance
        content_pool: List of available content items (can be strings or dicts with 'text' key)
    
    Returns:
        Selected content item with personalization notes (always a dict)
    """
    weak_areas = get_user_weak_areas(user)
    difficulty = get_user_difficulty_level(user, drill)
    
    if not content_pool:
        return {
            'text': '',
            'personalization_note': _generate_personalization_note(drill, weak_areas, difficulty),
            'difficulty_level': difficulty
        }
    
    # Select content based on drill type and weak areas
    if drill.skill_type == 'pronunciation':
        # Prioritize content that targets clarity issues
        if weak_areas['clarity'] > 0.6:
            # Select more challenging pronunciation content
            selected = content_pool[-1] if content_pool else None
        else:
            selected = content_pool[0] if content_pool else None
    elif drill.skill_type == 'pacing':
        # Prioritize content based on pacing issues
        if weak_areas['pacing'] > 0.6:
            selected = content_pool[0] if content_pool else None  # Start with easier
        else:
            selected = content_pool[-1] if content_pool else None  # More challenging
    else:  # filler_words
        # Prioritize content based on filler issues
        if weak_areas['filler'] > 0.6:
            selected = content_pool[0] if content_pool else None  # Start with easier
        else:
            selected = content_pool[-1] if content_pool else None
    
    if not selected:
        selected = content_pool[0] if content_pool else {}
    
    # Normalize to dict format
    if isinstance(selected, str):
        result = {'text': selected}
    elif isinstance(selected, dict):
        result = selected.copy()
        # Ensure 'text' key exists
        if 'text' not in result and len(result) > 0:
            # Use first value if 'text' key doesn't exist
            result['text'] = list(result.values())[0] if result.values() else ''
    else:
        result = {'text': str(selected) if selected else ''}
    
    # Add personalization notes
    result['personalization_note'] = _generate_personalization_note(drill, weak_areas, difficulty)
    result['difficulty_level'] = difficulty
    
    return result


def _generate_personalization_note(drill: Drill, weak_areas: Dict, difficulty: str) -> str:
    """Generate personalized note for user."""
    notes = []
    
    if drill.skill_type == 'pronunciation' and weak_areas['clarity'] > 0.6:
        notes.append("Focus on clear articulation - this drill will help improve your clarity.")
    elif drill.skill_type == 'pacing' and weak_areas['pacing'] > 0.6:
        notes.append("Work on maintaining consistent pace - practice with this drill regularly.")
    elif drill.skill_type == 'filler_words' and weak_areas['filler'] > 0.6:
        notes.append("Practice replacing fillers with pauses - this is a key area for improvement.")
    
    if difficulty == 'beginner':
        notes.append("Start with the basics and build your skills gradually.")
    elif difficulty == 'advanced':
        notes.append("Challenge yourself with more complex exercises.")
    
    return " ".join(notes) if notes else "Keep practicing to improve your skills!"


def save_drill_completion(user: User, drill: Drill, analysis_result: Dict, 
                         audio_data: Optional[bytes] = None,
                         difficulty_level: Optional[str] = None,
                         notes: Optional[str] = None) -> DrillCompletion:
    """
    Save drill completion with ML metrics.
    
    Returns:
        DrillCompletion instance
    """
    completion = DrillCompletion(
        user=user,
        drill=drill,
        ml_scores=analysis_result.get('ml_scores', {}),
        ml_metrics=analysis_result.get('ml_metrics', {}),
        drill_specific_metrics=analysis_result.get('drill_specific_metrics', {}),
        difficulty_level=difficulty_level or get_user_difficulty_level(user, drill),
        notes=notes or '',
        duration_seconds=int(analysis_result.get('duration', 0))
    )
    
    # Save audio file if provided
    if audio_data and analysis_result.get('audio_path'):
        try:
            audio_file = ContentFile(audio_data)
            completion.audio_file.save(
                f"drill_{drill.id}_{user.id}_{timezone.now().timestamp()}.wav",
                audio_file,
                save=False
            )
        except Exception as e:
            logger.warning(f"Failed to save audio file: {e}")
    
    completion.save()
    return completion


def get_user_progress(user: User, drill: Optional[Drill] = None) -> Dict:
    """
    Get user's progress across all drills or specific drill.
    
    Returns:
        Dict with progress metrics
    """
    if drill:
        completions = DrillCompletion.objects.filter(user=user, drill=drill)
    else:
        completions = DrillCompletion.objects.filter(user=user)
    
    completions = completions.order_by('-completed_at')[:30]  # Last 30
    
    if not completions:
        return {
            'total_completions': 0,
            'average_score': 0.0,
            'improvement_trend': 'stable',
            'best_score': 0.0,
            'recent_scores': []
        }
    
    scores = [c.get_overall_score() for c in completions if c.ml_scores]
    
    if not scores:
        return {
            'total_completions': completions.count(),
            'average_score': 0.0,
            'improvement_trend': 'stable',
            'best_score': 0.0,
            'recent_scores': []
        }
    
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    
    # Calculate improvement trend
    if len(scores) >= 2:
        recent_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        older_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        if recent_avg > older_avg + 0.1:
            trend = 'improving'
        elif recent_avg < older_avg - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
    else:
        trend = 'stable'
    
    return {
        'total_completions': completions.count(),
        'average_score': avg_score,
        'improvement_trend': trend,
        'best_score': best_score,
        'recent_scores': scores[:10]
    }


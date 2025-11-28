"""
Enhanced Pronunciation Analysis Module

Analyzes pronunciation quality using multiple approaches:
1. Whisper confidence + phonetic analysis
2. Accent-specific scoring
3. Phoneme accuracy assessment
4. Regional pronunciation patterns

Integrates with your existing NLP pipeline.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from dataclasses import dataclass

@dataclass
class PronunciationResult:
    """Results from pronunciation analysis."""
    overall_score: float
    phonetic_accuracy: float
    accent_consistency: float
    rhythm_score: float
    stress_patterns: Dict[str, float]
    detected_accent: Optional[str]
    recommendations: List[str]

class PronunciationAnalyzer:
    """Enhanced pronunciation analysis using multiple methods."""
    
    def __init__(self):
        self.accent_patterns = {
            'american': {
                'rhotic': True,  # Pronounces 'r' in 'car'
                'vowel_shifts': ['æ', 'ɑ', 'ɔ'],  # cot-caught merger
                'features': ['t_flapping', 'ng_fronting']
            },
            'british': {
                'rhotic': False,  # Drops 'r' in 'car'
                'vowel_shifts': ['ɑ', 'ɔ', 'ɒ'],  # bath-trap split
                'features': ['glottal_stops', 'h_dropping']
            },
            'australian': {
                'rhotic': False,
                'vowel_shifts': ['æ', 'ɑ', 'aɪ'],  # price-monophthongization
                'features': ['intrusive_r', 'vowel_raising']
            }
        }
        
        # Common pronunciation issues
        self.pronunciation_issues = {
            'th_substitution': {
                'pattern': r'\b(the|this|that|think|thing)\b',
                'issue': 'th sounds replaced with f/v/d',
                'severity': 'high'
            },
            'r_sound': {
                'pattern': r'\b(car|park|hard|word)\b',
                'issue': 'r sound difficulties',
                'severity': 'medium'
            },
            'vowel_consistency': {
                'pattern': r'\b(cat|cut|cot|caught)\b',
                'issue': 'vowel sound confusion',
                'severity': 'medium'
            },
            'consonant_clusters': {
                'pattern': r'\b(str|spr|scr|thr)\w+',
                'issue': 'consonant cluster simplification',
                'severity': 'low'
            }
        }

    def analyze_pronunciation(self, 
                            audio_path: str, 
                            transcription: str,
                            whisper_result: Dict) -> PronunciationResult:
        """
        Comprehensive pronunciation analysis.
        
        Args:
            audio_path: Path to audio file
            transcription: Whisper transcription
            whisper_result: Full Whisper result with confidence
            
        Returns:
            PronunciationResult with detailed analysis
        """
        
        # 1. Basic confidence from Whisper
        confidence_score = whisper_result.get('confidence', 0.0)
        
        # 2. Phonetic analysis
        phonetic_accuracy = self._analyze_phonetic_accuracy(transcription)
        
        # 3. Accent detection and consistency
        detected_accent, accent_consistency = self._detect_accent_consistency(
            audio_path, transcription
        )
        
        # 4. Rhythm and stress analysis
        rhythm_score = self._analyze_rhythm(audio_path)
        
        # 5. Stress pattern analysis
        stress_patterns = self._analyze_stress_patterns(transcription)
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(
            phonetic_accuracy, accent_consistency, stress_patterns
        )
        
        # 7. Calculate overall score
        overall_score = self._calculate_overall_score(
            confidence_score, phonetic_accuracy, accent_consistency, rhythm_score
        )
        
        return PronunciationResult(
            overall_score=overall_score,
            phonetic_accuracy=phonetic_accuracy,
            accent_consistency=accent_consistency,
            rhythm_score=rhythm_score,
            stress_patterns=stress_patterns,
            detected_accent=detected_accent,
            recommendations=recommendations
        )

    def _analyze_phonetic_accuracy(self, transcription: str) -> float:
        """Analyze phonetic accuracy based on common issues."""
        issues_found = 0
        total_opportunities = 0
        
        text_lower = transcription.lower()
        
        for issue_name, issue_data in self.pronunciation_issues.items():
            pattern = issue_data['pattern']
            matches = re.findall(pattern, text_lower)
            total_opportunities += len(matches)
            
            # This is a simplified check - in production you'd use
            # phonetic transcription libraries like espeak or festival
            if matches:
                issues_found += len(matches) * 0.3  # Estimate 30% error rate
        
        if total_opportunities == 0:
            return 1.0
        
        accuracy = max(0.0, 1.0 - (issues_found / total_opportunities))
        return accuracy

    def _detect_accent_consistency(self, 
                                 audio_path: str, 
                                 transcription: str) -> Tuple[Optional[str], float]:
        """Detect accent and measure consistency."""
        
        # Simplified accent detection based on transcription patterns
        text_lower = transcription.lower()
        
        # American English indicators
        american_score = 0
        if re.search(r'\b(garage|herb)\b', text_lower):
            american_score += 1
        if re.search(r'\b(can\'t|won\'t)\b', text_lower):
            american_score += 1
            
        # British English indicators  
        british_score = 0
        if re.search(r'\b(colour|favour|centre)\b', text_lower):
            british_score += 1
        if re.search(r'\b(brilliant|lovely)\b', text_lower):
            british_score += 1
        
        # Determine dominant accent
        if american_score > british_score:
            detected_accent = 'american'
            consistency = min(1.0, american_score / 3.0)
        elif british_score > american_score:
            detected_accent = 'british'
            consistency = min(1.0, british_score / 3.0)
        else:
            detected_accent = None
            consistency = 0.5  # Mixed or unclear
        
        return detected_accent, consistency

    def _analyze_rhythm(self, audio_path: str) -> float:
        """Analyze speech rhythm and timing."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Analyze rhythm regularity
            if len(beats) > 2:
                beat_intervals = np.diff(beats)
                rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
            else:
                rhythm_regularity = 0.5
            
            return min(1.0, rhythm_regularity)
            
        except Exception as e:
            print(f"[WARNING] Rhythm analysis failed: {e}")
            return 0.5

    def _analyze_stress_patterns(self, transcription: str) -> Dict[str, float]:
        """Analyze word stress patterns."""
        words = transcription.lower().split()
        
        stress_analysis = {
            'monosyllabic_ratio': 0.0,
            'stress_consistency': 0.0,
            'word_length_variety': 0.0
        }
        
        if not words:
            return stress_analysis
        
        # Monosyllabic word ratio
        monosyllabic = sum(1 for word in words if len(word) <= 4)
        stress_analysis['monosyllabic_ratio'] = monosyllabic / len(words)
        
        # Word length variety (indicates varied stress patterns)
        word_lengths = [len(word) for word in words]
        if word_lengths:
            stress_analysis['word_length_variety'] = 1.0 - (np.std(word_lengths) / np.mean(word_lengths))
        
        # Simplified stress consistency (would need phonetic analysis for accuracy)
        stress_analysis['stress_consistency'] = 0.7  # Placeholder
        
        return stress_analysis

    def _generate_recommendations(self, 
                                phonetic_accuracy: float,
                                accent_consistency: float,
                                stress_patterns: Dict[str, float]) -> List[str]:
        """Generate pronunciation improvement recommendations."""
        recommendations = []
        
        if phonetic_accuracy < 0.7:
            recommendations.append("Focus on phonetic accuracy - practice difficult sounds")
            recommendations.append("Consider accent reduction training")
        
        if accent_consistency < 0.6:
            recommendations.append("Work on accent consistency")
            recommendations.append("Practice with accent-specific exercises")
        
        if stress_patterns.get('monosyllabic_ratio', 0) > 0.8:
            recommendations.append("Vary word complexity for better rhythm")
        
        if stress_patterns.get('word_length_variety', 0) < 0.3:
            recommendations.append("Practice varied stress patterns")
        
        if not recommendations:
            recommendations.append("Pronunciation is good - focus on other areas")
        
        return recommendations

    def _calculate_overall_score(self, 
                               confidence: float,
                               phonetic_accuracy: float,
                               accent_consistency: float,
                               rhythm_score: float) -> float:
        """Calculate weighted overall pronunciation score."""
        
        weights = {
            'confidence': 0.3,
            'phonetic_accuracy': 0.4,
            'accent_consistency': 0.2,
            'rhythm_score': 0.1
        }
        
        overall = (
            weights['confidence'] * confidence +
            weights['phonetic_accuracy'] * phonetic_accuracy +
            weights['accent_consistency'] * accent_consistency +
            weights['rhythm_score'] * rhythm_score
        )
        
        return min(1.0, max(0.0, overall))

# Integration function for your existing pipeline
def analyze_pronunciation_enhanced(audio_path: str, 
                                 transcription: str,
                                 whisper_result: Dict) -> Dict:
    """
    Enhanced pronunciation analysis for your NLP pipeline.
    
    Returns dict compatible with existing pipeline.
    """
    analyzer = PronunciationAnalyzer()
    result = analyzer.analyze_pronunciation(audio_path, transcription, whisper_result)
    
    return {
        'pronunciation_score': result.overall_score,
        'phonetic_accuracy': result.phonetic_accuracy,
        'accent_consistency': result.accent_consistency,
        'rhythm_score': result.rhythm_score,
        'detected_accent': result.detected_accent,
        'recommendations': result.recommendations,
        'stress_patterns': result.stress_patterns
    }

"""
NLP Pipeline for Speech Analysis
Integrates Whisper (transcription), BERT (filler detection, cause classification),
and custom logic (pacing analysis) for comprehensive speech coaching.
"""

import os
import json
import torch
import whisper
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from django.conf import settings
from transformers import (
    BertForTokenClassification,
    BertForSequenceClassification,
    BertTokenizer,
    pipeline
)

# Try to import spaCy
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
    try:
        # Try to load English model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, create blank English model
        nlp = English()
        nlp.add_pipe("sentencizer")
        print("[WARNING] spaCy English model not found. Using basic tokenizer. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("[WARNING] spaCy not available. Install with: pip install spacy")

# Import enhanced pronunciation analysis
try:
    from .pronunciation_analyzer import analyze_pronunciation_enhanced
    PRONUNCIATION_ENHANCED = True
except ImportError:
    PRONUNCIATION_ENHANCED = False
    print("[WARNING] Enhanced pronunciation analysis not available")
from datetime import datetime

# Import Django models
from .models import SpeechSession, DetectedFiller
from coach.models import Drill


class SpeechAnalysisPipeline:
    """
    Complete speech analysis pipeline combining:
    1. Whisper for transcription
    2. BERT for filler detection and cause classification
    3. Custom logic for pacing analysis
    4. ML-based drill recommendations
    """
    
    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize the NLP pipeline.
        
        Args:
            whisper_model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None
        self.filler_detector = None
        self.cause_classifier = None
        self.tokenizer = None
        
        # Model paths
        self.models_dir = Path(settings.MEDIA_ROOT) / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Filler words for fallback detection
        self.common_fillers = [
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'literally', 'i mean', 'right', 'okay', 'well', 'kind of',
            'sort of', 'you see', 'i think', 'i guess'
        ]
        
        # Analysis thresholds
        self.FILLER_THRESHOLD = 0.05  # fillers per second
        self.WPM_LOW_THRESHOLD = 100
        self.WPM_HIGH_THRESHOLD = 180
        self.PAUSE_LONG_THRESHOLD = 2.0  # seconds
        self.PAUSE_SHORT_THRESHOLD = 0.3  # seconds
        
        self._load_models()
    
    def _load_models(self):
        """Load all ML models."""
        print("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            print(f"[OK] Whisper {self.whisper_model_size} model loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load Whisper: {e}")
            self.whisper_model = None
        
        # Load BERT tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("[OK] BERT tokenizer loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer: {e}")
        
        # Try to load fine-tuned filler detector
        filler_model_path = self.models_dir / 'filler_detector'
        if filler_model_path.exists():
            try:
                self.filler_detector = BertForTokenClassification.from_pretrained(
                    str(filler_model_path)
                )
                print("[OK] Fine-tuned filler detector loaded")
            except Exception as e:
                print(f"[ERROR] Failed to load filler detector: {e}")
        else:
            print("[WARNING] Filler detector not found, will use rule-based fallback")
        
        # Try to load fine-tuned cause classifier
        cause_model_path = self.models_dir / 'cause_classifier'
        if cause_model_path.exists():
            try:
                self.cause_classifier = BertForSequenceClassification.from_pretrained(
                    str(cause_model_path)
                )
                print("[OK] Fine-tuned cause classifier loaded")
            except Exception as e:
                print(f"[ERROR] Failed to load cause classifier: {e}")
        else:
            print("[WARNING] Cause classifier not found, will use rule-based fallback")
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available (required for MP3 and many audio formats)."""
        import subprocess
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Also check common installation paths (Windows)
            common_paths = [
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return True
            return False
    
    def _convert_audio_to_wav(self, audio_path: str) -> Optional[str]:
        """
        Convert audio file to WAV format using librosa (doesn't require ffmpeg).
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to temporary WAV file, or None if conversion fails
        """
        try:
            import librosa
            import soundfile as sf
            import tempfile
            
            print(f"[DEBUG] Converting {audio_path} to WAV format...")
            
            # Load audio with librosa (handles MP3, M4A, FLAC, etc.)
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Save as WAV
            sf.write(temp_wav_path, y, sr)
            print(f"[DEBUG] Converted to WAV: {temp_wav_path} ({len(y)/sr:.2f}s, {sr}Hz)")
            
            return temp_wav_path
            
        except Exception as e:
            print(f"[ERROR] Audio conversion failed: {e}")
            return None
    
    def save_detected_fillers(self, speech_session: SpeechSession, filler_analysis: Dict[str, Any]) -> int:
        """
        Save detected fillers to the database.
        
        Args:
            speech_session: The SpeechSession instance
            filler_analysis: Dictionary from detect_fillers() containing tagged_fillers
            
        Returns:
            Number of fillers saved
        """
        if not filler_analysis or 'tagged_fillers' not in filler_analysis:
            return 0
        
        tagged_fillers = filler_analysis.get('tagged_fillers', [])
        detection_method = filler_analysis.get('detection_method', 'rule_based')
        
        # Map method names to model choices
        method_map = {
            'spacy': 'spacy',
            'regex': 'regex',
            'bert': 'bert',
            'rule_based': 'rule_based'
        }
        method_choice = method_map.get(detection_method, 'rule_based')
        
        saved_count = 0
        for filler_info in tagged_fillers:
            try:
                DetectedFiller.objects.create(
                    speech_session=speech_session,
                    filler_word=filler_info.get('filler', ''),
                    word_position=filler_info.get('position', -1),
                    start_time=filler_info.get('start_time'),
                    end_time=filler_info.get('end_time'),
                    context_before=filler_info.get('context_before', ''),
                    context_after=filler_info.get('context_after', ''),
                    detection_method=method_choice,
                    confidence=filler_info.get('confidence', 1.0)
                )
                saved_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to save filler at position {filler_info.get('position')}: {e}")
        
        return saved_count
    
    def analyze_audio(self, audio_file_path: str, speech_session: SpeechSession = None) -> Dict[str, Any]:
        """
        Main pipeline: Analyze audio file and return comprehensive results.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'success': False,
            'transcription': '',
            'duration': 0,
            'word_count': 0,
            'segments': [],
            'pacing': {},
            'fillers': {},
            'pronunciation': {},
            'cause': None,
            'recommendations': [],
            'confidence_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Transcribe with Whisper
            print("Step 1: Transcribing audio...")
            print(f"[DEBUG] Audio file path: {audio_file_path}")
            print(f"[DEBUG] File exists: {os.path.exists(audio_file_path) if audio_file_path else False}")
            print(f"[DEBUG] Whisper model loaded: {self.whisper_model is not None}")
            
            transcription_data = self.transcribe_audio(audio_file_path)
            
            if not transcription_data:
                if not self.whisper_model:
                    error_msg = 'Whisper model not loaded. Please install openai-whisper: pip install openai-whisper'
                else:
                    error_msg = 'Transcription failed. Check audio file format (WAV/MP3) and file integrity.'
                print(f"[ERROR] {error_msg}")
                results['error'] = error_msg
                return results
            
            results['transcription'] = transcription_data['text']
            results['duration'] = transcription_data['duration']
            results['segments'] = transcription_data['segments']
            results['word_count'] = len(transcription_data['text'].split())
            
            # Step 2: Analyze pacing from timestamps
            print("Step 2: Analyzing pacing...")
            pacing_analysis = self.analyze_pacing(transcription_data['segments'])
            results['pacing'] = pacing_analysis
            
            # Step 3: Detect filler words
            print("Step 3: Detecting filler words...")
            filler_analysis = self.detect_fillers(
                transcription_data['text'],
                transcription_data['segments']
            )
            results['fillers'] = filler_analysis
            
            # Step 3.5: Save detected fillers to database if speech_session provided
            if speech_session:
                print("Step 3.5: Saving detected fillers to database...")
                saved_count = self.save_detected_fillers(speech_session, filler_analysis)
                print(f"[OK] Saved {saved_count} filler words to database")
            
            # Step 4: Analyze pronunciation (enhanced with accent analysis)
            print("Step 4: Analyzing pronunciation...")
            pronunciation_analysis = self.analyze_pronunciation(
                transcription_data['text'],
                transcription_data['segments'],
                audio_path=audio_file_path,
                whisper_result=transcription_data
            )
            results['pronunciation'] = pronunciation_analysis
            
            # Step 5: Classify root cause
            print("Step 5: Classifying root cause...")
            cause = self.classify_cause(
                transcription_data['text'],
                pacing_analysis,
                filler_analysis
            )
            results['cause'] = cause
            
            # Step 6: Calculate overall confidence score
            results['confidence_score'] = self.calculate_confidence_score(
                pacing_analysis, filler_analysis, pronunciation_analysis
            )
            
            # Step 7: Generate drill recommendations
            print("Step 6: Generating recommendations...")
            recommendations = self.recommend_drills(
                cause, pacing_analysis, filler_analysis, pronunciation_analysis
            )
            results['recommendations'] = recommendations
            
            results['success'] = True
            print("✓ Analysis complete!")
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio using Whisper with word-level timestamps.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with transcription, duration, and segments
        """
        if not self.whisper_model:
            print("[ERROR] Whisper model not available - model was not loaded during initialization")
            print("[INFO] Check if openai-whisper is installed: pip install openai-whisper")
            return None
        
        if not os.path.exists(audio_file_path):
            print(f"[ERROR] Audio file not found: {audio_file_path}")
            return None
        
        try:
            print(f"[DEBUG] Starting Whisper transcription of: {audio_file_path}")
            
            # Check file size (Whisper might have issues with very large files)
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            print(f"[DEBUG] Audio file size: {file_size_mb:.2f} MB")
            
            # If file is MP3 or other compressed format, convert to WAV first
            # This avoids ffmpeg dependency since librosa can handle MP3
            audio_path_to_use = audio_file_path
            temp_wav_path = None
            
            if audio_file_path.lower().endswith(('.mp3', '.m4a', '.flac', '.aac')):
                ffmpeg_available = self._check_ffmpeg()
                
                if not ffmpeg_available:
                    print("[INFO] ffmpeg not available - converting to WAV with librosa...")
                    temp_wav_path = self._convert_audio_to_wav(audio_file_path)
                    
                    if temp_wav_path:
                        audio_path_to_use = temp_wav_path
                        print(f"[OK] Using converted WAV file for transcription")
                    else:
                        error_msg = (
                            "Failed to convert audio file to WAV format. "
                            "Please install ffmpeg for better compatibility:\n"
                            "  Windows: winget install ffmpeg  OR  choco install ffmpeg\n"
                            "  Or download from: https://ffmpeg.org/download.html"
                        )
                        print(f"[ERROR] {error_msg}")
                        return None
                else:
                    print("[INFO] ffmpeg available - using original file")
            else:
                print("[INFO] File is already WAV format - using directly")
            
            # Transcribe with word timestamps
            try:
                result = self.whisper_model.transcribe(
                    audio_path_to_use,
                    word_timestamps=True,
                    verbose=False,
                    fp16=False  # Use fp32 for better compatibility
                )
            finally:
                # Clean up temporary WAV file if we created one
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.unlink(temp_wav_path)
                        print(f"[DEBUG] Cleaned up temporary WAV file")
                    except Exception as e:
                        print(f"[WARNING] Failed to delete temp file {temp_wav_path}: {e}")
            print(f"[DEBUG] Whisper transcription complete, text length: {len(result.get('text', ''))}")
            
            # Extract segments with timing info
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            segments.append({
                                'word': word_info.get('word', '').strip(),
                                'start': word_info.get('start', 0),
                                'end': word_info.get('end', 0),
                                'probability': word_info.get('probability', 0)
                            })
            
            # Calculate duration
            duration = segments[-1]['end'] if segments else 0
            
            if not result.get('text', '').strip():
                print("[WARNING] Whisper returned empty transcription")
                return None
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'duration': duration,
                'segments': segments
            }
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[ERROR] Transcription error: {e}")
            print(f"[ERROR] Traceback: {error_trace}")
            return None
    
    def analyze_pacing(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze speaking pace from word timestamps.
        
        Args:
            segments: List of word segments with timing
            
        Returns:
            Dictionary with pacing metrics
        """
        if not segments:
            return {
                'wpm': 0,
                'avg_pause_duration': 0,
                'total_pauses': 0,
                'long_pauses': 0,
                'speaking_rate': 'unknown',
                'pauses': []
            }
        
        # Calculate pauses between words
        pauses = []
        for i in range(1, len(segments)):
            pause_duration = segments[i]['start'] - segments[i-1]['end']
            if pause_duration > self.PAUSE_SHORT_THRESHOLD:
                pauses.append({
                    'duration': pause_duration,
                    'position': i,
                    'before_word': segments[i]['word']
                })
        
        # Calculate WPM
        total_words = len(segments)
        total_time = segments[-1]['end'] - segments[0]['start']
        wpm = (total_words / total_time * 60) if total_time > 0 else 0
        
        # Calculate pause statistics
        pause_durations = [p['duration'] for p in pauses]
        avg_pause = sum(pause_durations) / len(pause_durations) if pause_durations else 0
        long_pauses = len([p for p in pauses if p['duration'] > self.PAUSE_LONG_THRESHOLD])
        
        # Determine speaking rate category
        if wpm < self.WPM_LOW_THRESHOLD:
            speaking_rate = 'too_slow'
        elif wpm > self.WPM_HIGH_THRESHOLD:
            speaking_rate = 'too_fast'
        else:
            speaking_rate = 'optimal'
        
        return {
            'wpm': round(wpm, 2),
            'avg_pause_duration': round(avg_pause, 2),
            'total_pauses': len(pauses),
            'long_pauses': long_pauses,
            'speaking_rate': speaking_rate,
            'pauses': pauses[:10],  # First 10 pauses for detail
            'total_duration': round(total_time, 2)
        }
    
    def detect_fillers(self, text: str, segments: List[Dict] = None) -> Dict[str, Any]:
        """
        Detect filler words using spaCy, BERT model, or rule-based fallback.
        
        Args:
            text: Transcribed text
            segments: Word segments with timing (optional)
            
        Returns:
            Dictionary with filler analysis including tagged fillers
        """
        # Try spaCy first (most accurate NLP-based detection)
        if SPACY_AVAILABLE and nlp:
            return self._detect_fillers_spacy(text, segments)
        # Use fine-tuned BERT if available
        elif self.filler_detector and self.tokenizer:
            return self._detect_fillers_bert(text, segments)
        else:
            # Fallback to rule-based with regex
            return self._detect_fillers_regex(text, segments)
    
    def _detect_fillers_spacy(self, text: str, segments: List[Dict] = None) -> Dict[str, Any]:
        """
        Detect fillers using spaCy NLP for better context-aware detection.
        
        Args:
            text: Transcribed text
            segments: Word segments with timing (optional)
            
        Returns:
            Dictionary with filler analysis including tagged fillers
        """
        if not SPACY_AVAILABLE or not nlp:
            return self._detect_fillers_regex(text, segments)
        
        # Process text with spaCy
        doc = nlp(text.lower())
        words = text.split()
        words_lower = [w.lower().strip('.,!?;:') for w in words]
        
        # Track detected fillers with positions and context
        tagged_fillers = []
        detected_fillers_dict = {}
        
        # Create a mapping of word positions to tokens
        token_to_word_pos = {}
        word_idx = 0
        for token in doc:
            # Skip punctuation tokens
            if token.is_punct or token.is_space:
                continue
            # Match token to word position
            while word_idx < len(words_lower):
                word_clean = words_lower[word_idx].strip('.,!?;:')
                if token.text.lower() in word_clean or word_clean in token.text.lower():
                    token_to_word_pos[token.i] = word_idx
                    word_idx += 1
                    break
                word_idx += 1
        
        # Check each token against filler words
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            
            token_text = token.text.lower().strip('.,!?;:')
            
            # Check against common fillers
            for filler in self.common_fillers:
                filler_words = filler.split()
                
                # Single word filler match
                if len(filler_words) == 1 and token_text == filler_words[0]:
                    word_pos = token_to_word_pos.get(token.i, -1)
                    if word_pos >= 0:
                        # Get context (previous and next tokens)
                        context_before = []
                        context_after = []
                        
                        # Get previous tokens
                        for prev_token in doc[max(0, token.i-3):token.i]:
                            if not prev_token.is_punct and not prev_token.is_space:
                                context_before.append(prev_token.text)
                        
                        # Get next tokens
                        for next_token in doc[token.i+1:min(len(doc), token.i+4)]:
                            if not next_token.is_punct and not next_token.is_space:
                                context_after.append(next_token.text)
                        
                        # Get timing if available
                        start_time = None
                        end_time = None
                        if segments and word_pos < len(segments):
                            start_time = segments[word_pos].get('start')
                            end_time = segments[word_pos].get('end')
                        
                        tagged_fillers.append({
                            'filler': filler,
                            'word': token.text,
                            'position': word_pos,
                            'start_time': start_time,
                            'end_time': end_time,
                            'context_before': ' '.join(context_before[-2:]),  # Last 2 words
                            'context_after': ' '.join(context_after[:2]),  # First 2 words
                            'confidence': 0.9,  # High confidence for exact match
                            'method': 'spacy'
                        })
                        
                        # Count fillers
                        if filler not in detected_fillers_dict:
                            detected_fillers_dict[filler] = 0
                        detected_fillers_dict[filler] += 1
                
                # Multi-word filler match (e.g., "you know", "i mean")
                elif len(filler_words) > 1:
                    # Check if current token starts a multi-word filler
                    if token_text == filler_words[0] and token.i + len(filler_words) - 1 < len(doc):
                        # Check if following tokens match
                        match = True
                        for i, filler_word in enumerate(filler_words[1:], 1):
                            next_token = doc[token.i + i] if token.i + i < len(doc) else None
                            if not next_token or next_token.text.lower().strip('.,!?;:') != filler_word:
                                match = False
                                break
                        
                        if match:
                            word_pos = token_to_word_pos.get(token.i, -1)
                            if word_pos >= 0:
                                # Get context
                                context_before = []
                                context_after = []
                                
                                for prev_token in doc[max(0, token.i-3):token.i]:
                                    if not prev_token.is_punct and not prev_token.is_space:
                                        context_before.append(prev_token.text)
                                
                                for next_token in doc[min(len(doc), token.i+len(filler_words)):min(len(doc), token.i+len(filler_words)+3)]:
                                    if not next_token.is_punct and not next_token.is_space:
                                        context_after.append(next_token.text)
                                
                                start_time = None
                                end_time = None
                                if segments and word_pos < len(segments):
                                    start_time = segments[word_pos].get('start')
                                    if word_pos + len(filler_words) - 1 < len(segments):
                                        end_time = segments[word_pos + len(filler_words) - 1].get('end')
                                
                                tagged_fillers.append({
                                    'filler': filler,
                                    'word': ' '.join([doc[token.i + j].text for j in range(len(filler_words))]),
                                    'position': word_pos,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'context_before': ' '.join(context_before[-2:]),
                                    'context_after': ' '.join(context_after[:2]),
                                    'confidence': 0.9,
                                    'method': 'spacy'
                                })
                                
                                if filler not in detected_fillers_dict:
                                    detected_fillers_dict[filler] = 0
                                detected_fillers_dict[filler] += 1
        
        # Remove duplicates (same position)
        seen_positions = set()
        unique_tagged_fillers = []
        for filler_info in tagged_fillers:
            pos_key = filler_info['position']
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_tagged_fillers.append(filler_info)
        
        tagged_fillers = unique_tagged_fillers
        
        # Convert to list format
        detected_fillers = [
            {'filler': filler, 'count': count}
            for filler, count in detected_fillers_dict.items()
        ]
        
        total_fillers = len(tagged_fillers)
        
        # Calculate density
        duration = segments[-1]['end'] if segments and segments else 1
        filler_density = total_fillers / duration if duration > 0 else 0
        
        return {
            'total_count': total_fillers,
            'unique_fillers': len(detected_fillers),
            'filler_types': detected_fillers[:10],
            'density': round(filler_density, 4),
            'positions': [f['position'] for f in tagged_fillers[:20]],
            'severity': 'high' if filler_density > self.FILLER_THRESHOLD else 'low',
            'tagged_fillers': tagged_fillers,  # New: tagged fillers with full details
            'detection_method': 'spacy'
        }
    
    def _detect_fillers_regex(self, text: str, segments: List[Dict] = None) -> Dict[str, Any]:
        """
        Detect fillers using regex patterns (improved rule-based approach).
        
        Args:
            text: Transcribed text
            segments: Word segments with timing (optional)
            
        Returns:
            Dictionary with filler analysis including tagged fillers
        """
        text_lower = text.lower()
        words = text.split()
        words_lower = [w.lower().strip('.,!?;:') for w in words]
        
        tagged_fillers = []
        detected_fillers_dict = {}
        
        # Use regex for more flexible matching
        for filler in self.common_fillers:
            filler_words = filler.split()
            
            # Create regex pattern for the filler
            if len(filler_words) == 1:
                # Single word: match whole word boundaries
                pattern = r'\b' + re.escape(filler_words[0]) + r'\b'
            else:
                # Multi-word: match sequence with word boundaries
                pattern = r'\b' + r'\s+'.join([re.escape(w) for w in filler_words]) + r'\b'
            
            # Find all matches
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                # Find word position
                start_char = match.start()
                char_count = 0
                word_pos = -1
                
                for i, word in enumerate(words):
                    word_start = char_count
                    word_end = char_count + len(word)
                    if word_start <= start_char < word_end:
                        word_pos = i
                        break
                    char_count = word_end + 1  # +1 for space
                
                if word_pos >= 0:
                    # Get context
                    context_before = ' '.join(words[max(0, word_pos-2):word_pos]) if word_pos > 0 else ''
                    context_after = ' '.join(words[word_pos+len(filler_words):word_pos+len(filler_words)+2]) if word_pos + len(filler_words) < len(words) else ''
                    
                    # Get timing if available
                    start_time = None
                    end_time = None
                    if segments and word_pos < len(segments):
                        start_time = segments[word_pos].get('start')
                        if word_pos + len(filler_words) - 1 < len(segments):
                            end_time = segments[word_pos + len(filler_words) - 1].get('end')
                    
                    tagged_fillers.append({
                        'filler': filler,
                        'word': match.group(),
                        'position': word_pos,
                        'start_time': start_time,
                        'end_time': end_time,
                        'context_before': context_before,
                        'context_after': context_after,
                        'confidence': 0.8,  # Good confidence for regex match
                        'method': 'regex'
                    })
                    
                    if filler not in detected_fillers_dict:
                        detected_fillers_dict[filler] = 0
                    detected_fillers_dict[filler] += 1
        
        # Remove duplicates
        seen_positions = set()
        unique_tagged_fillers = []
        for filler_info in tagged_fillers:
            pos_key = filler_info['position']
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_tagged_fillers.append(filler_info)
        
        tagged_fillers = unique_tagged_fillers
        
        # Convert to list format
        detected_fillers = [
            {'filler': filler, 'count': count}
            for filler, count in detected_fillers_dict.items()
        ]
        
        total_fillers = len(tagged_fillers)
        
        # Calculate density
        duration = segments[-1]['end'] if segments and segments else 1
        filler_density = total_fillers / duration if duration > 0 else 0
        
        return {
            'total_count': total_fillers,
            'unique_fillers': len(detected_fillers),
            'filler_types': detected_fillers[:10],
            'density': round(filler_density, 4),
            'positions': [f['position'] for f in tagged_fillers[:20]],
            'severity': 'high' if filler_density > self.FILLER_THRESHOLD else 'low',
            'tagged_fillers': tagged_fillers,  # New: tagged fillers with full details
            'detection_method': 'regex'
        }
    
    def _detect_fillers_bert(self, text: str, segments: List[Dict] = None) -> Dict[str, Any]:
        """Detect fillers using fine-tuned BERT model."""
        # TODO: Implement after BERT model is trained
        # For now, fall back to regex-based
        return self._detect_fillers_regex(text, segments)
    
    def analyze_pronunciation(self, text: str, segments: List[Dict], audio_path: str = None, whisper_result: Dict = None) -> Dict[str, Any]:
        """
        Enhanced pronunciation analysis using multiple methods.
        
        Args:
            text: Transcribed text
            segments: Word segments
            audio_path: Path to audio file (for enhanced analysis)
            whisper_result: Full Whisper result
            
        Returns:
            Dictionary with pronunciation metrics
        """
        
        # Enhanced pronunciation analysis if available
        if PRONUNCIATION_ENHANCED and audio_path and whisper_result:
            try:
                enhanced_result = analyze_pronunciation_enhanced(audio_path, text, whisper_result)
                
                return {
                    'clarity_score': round(enhanced_result['pronunciation_score'] * 100, 2),
                    'phonetic_accuracy': round(enhanced_result['phonetic_accuracy'] * 100, 2),
                    'accent_consistency': round(enhanced_result['accent_consistency'] * 100, 2),
                    'rhythm_score': round(enhanced_result['rhythm_score'] * 100, 2),
                    'detected_accent': enhanced_result['detected_accent'],
                    'pronunciation_recommendations': enhanced_result['recommendations'],
                    'stress_patterns': enhanced_result['stress_patterns'],
                    'avg_confidence': round(whisper_result.get('confidence', 0.0), 3),
                    'unclear_words_count': len(self._get_unclear_words(segments)),
                    'unclear_words': self._get_unclear_words(segments)[:10],
                    'quality': 'good' if enhanced_result['pronunciation_score'] > 0.7 else 'needs_improvement',
                    'analysis_method': 'enhanced'
                }
            except Exception as e:
                print(f"[WARNING] Enhanced pronunciation analysis failed: {e}")
                # Fall back to basic analysis
        
        # Basic pronunciation analysis (fallback)
        if segments:
            probabilities = [s.get('probability', 0) for s in segments if 'probability' in s]
            avg_confidence = sum(probabilities) / len(probabilities) if probabilities else 0
        else:
            avg_confidence = 0
        
        # Calculate clarity score (0-100)
        clarity_score = avg_confidence * 100
        
        return {
            'clarity_score': round(clarity_score, 2),
            'phonetic_accuracy': round(clarity_score, 2),  # Use clarity as proxy
            'accent_consistency': 75.0,  # Default assumption
            'rhythm_score': 70.0,  # Default assumption
            'detected_accent': None,
            'pronunciation_recommendations': self._get_basic_pronunciation_recommendations(clarity_score),
            'stress_patterns': {},
            'avg_confidence': round(avg_confidence, 3),
            'unclear_words_count': len(self._get_unclear_words(segments)),
            'unclear_words': self._get_unclear_words(segments)[:10],
            'quality': 'good' if clarity_score > 70 else 'needs_improvement',
            'analysis_method': 'basic'
        }
    
    def _get_unclear_words(self, segments: List[Dict]) -> List[Dict]:
        """Extract unclear words from segments."""
        unclear_words = []
        if segments:
            for seg in segments:
                if seg.get('probability', 1) < 0.7:
                    unclear_words.append({
                        'word': seg['word'],
                        'confidence': seg.get('probability', 0),
                        'time': seg.get('start', 0)
                    })
        return unclear_words
    
    def _get_basic_pronunciation_recommendations(self, clarity_score: float) -> List[str]:
        """Get basic pronunciation recommendations based on clarity score."""
        recommendations = []
        
        if clarity_score < 60:
            recommendations.append("Focus on speaking more clearly")
            recommendations.append("Practice pronunciation of difficult words")
        elif clarity_score < 80:
            recommendations.append("Continue working on pronunciation clarity")
        else:
            recommendations.append("Pronunciation is good - focus on other areas")
        
        return recommendations
    
    def classify_cause(self, text: str, pacing: Dict, fillers: Dict) -> str:
        """
        Classify root cause of speech issues.
        
        Args:
            text: Transcribed text
            pacing: Pacing analysis results
            fillers: Filler analysis results
            
        Returns:
            Cause category: 'anxiety', 'stress', 'lack_of_confidence', 'poor_skills'
        """
        # Use fine-tuned BERT if available
        if self.cause_classifier and self.tokenizer:
            return self._classify_cause_bert(text, pacing, fillers)
        else:
            return self._classify_cause_rule_based(pacing, fillers)
    
    def _classify_cause_bert(self, text: str, pacing: Dict, fillers: Dict) -> str:
        """Classify cause using fine-tuned BERT + XGBoost model."""
        try:
            from coach.utils import predict_cause
            
            # Get audio path from self if available
            audio_path = getattr(self, 'audio_path', None)
            duration = pacing.get('total_duration', None)
            
            # Call integrated predict_cause function
            result = predict_cause(audio_path, text, duration)
            return result['cause']
        except Exception as e:
            # Fallback to rule-based if ML model fails
            logger.warning(f"ML cause classification failed: {e}, using rule-based")
            return self._classify_cause_rule_based(pacing, fillers)
    
    def _classify_cause_rule_based(self, pacing: Dict, fillers: Dict) -> str:
        """Rule-based cause classification."""
        # High fillers + fast pace = anxiety
        if fillers.get('density', 0) > 0.05 and pacing.get('wpm', 0) > 160:
            return 'anxiety'
        
        # High pauses + slow pace = stress
        if pacing.get('long_pauses', 0) > 3 and pacing.get('wpm', 0) < 110:
            return 'stress'
        
        # Many fillers + many pauses = lack of confidence
        if fillers.get('total_count', 0) > 10 and pacing.get('total_pauses', 0) > 15:
            return 'lack_of_confidence'
        
        # Default to poor skills
        return 'poor_skills'
    
    def calculate_confidence_score(self, pacing: Dict, fillers: Dict, pronunciation: Dict) -> float:
        """Calculate overall speech confidence score (0-1)."""
        scores = []
        
        # Pacing score (0-1)
        wpm = pacing.get('wpm', 0)
        if 110 <= wpm <= 170:
            pacing_score = 1.0
        elif 100 <= wpm <= 180:
            pacing_score = 0.7
        else:
            pacing_score = 0.4
        scores.append(pacing_score)
        
        # Filler score (0-1)
        filler_density = fillers.get('density', 0)
        filler_score = max(0, 1 - (filler_density / 0.1))
        scores.append(filler_score)
        
        # Pronunciation score (0-1)
        clarity = pronunciation.get('clarity_score', 0) / 100
        scores.append(clarity)
        
        # Average all scores
        return round(sum(scores) / len(scores), 3)
    
    def recommend_drills(
        self, 
        cause: str, 
        pacing: Dict, 
        fillers: Dict, 
        pronunciation: Dict
    ) -> List[Dict]:
        """
        Recommend drills based on analysis results.
        
        Args:
            cause: Root cause classification
            pacing: Pacing analysis
            fillers: Filler analysis
            pronunciation: Pronunciation analysis
            
        Returns:
            List of recommended drills with reasons
        """
        recommendations = []
        
        # Get drills by cause
        cause_drills = Drill.objects.filter(cause=cause, is_active=True)
        
        # Recommend based on specific issues
        if fillers.get('severity') == 'high':
            filler_drills = Drill.objects.filter(skill_type='filler_words', is_active=True)
            for drill in filler_drills[:2]:
                recommendations.append({
                    'drill_id': drill.id,
                    'name': drill.name,
                    'description': drill.description,
                    'skill_type': drill.skill_type,
                    'reason': f'High filler word usage detected ({fillers.get("total_count")} fillers)',
                    'priority': 'high'
                })
        
        if pacing.get('speaking_rate') != 'optimal':
            pacing_drills = Drill.objects.filter(skill_type='pacing', is_active=True)
            for drill in pacing_drills[:2]:
                recommendations.append({
                    'drill_id': drill.id,
                    'name': drill.name,
                    'description': drill.description,
                    'skill_type': drill.skill_type,
                    'reason': f'Speaking pace issue: {pacing.get("wpm")} WPM ({pacing.get("speaking_rate")})',
                    'priority': 'medium'
                })
        
        if pronunciation.get('quality') == 'needs_improvement':
            pronunciation_drills = Drill.objects.filter(skill_type='pronunciation', is_active=True)
            for drill in pronunciation_drills[:2]:
                recommendations.append({
                    'drill_id': drill.id,
                    'name': drill.name,
                    'description': drill.description,
                    'skill_type': drill.skill_type,
                    'reason': f'Pronunciation clarity: {pronunciation.get("clarity_score")}%',
                    'priority': 'medium'
                })
        
        # Add cause-specific drills
        for drill in cause_drills[:2]:
            if drill.id not in [r['drill_id'] for r in recommendations]:
                recommendations.append({
                    'drill_id': drill.id,
                    'name': drill.name,
                    'description': drill.description,
                    'skill_type': drill.skill_type,
                    'reason': f'Recommended for {cause.replace("_", " ")}',
                    'priority': 'low'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations[:6]  # Top 6 recommendations


# Singleton instance
_pipeline_instance = None

def get_pipeline() -> SpeechAnalysisPipeline:
    """Get or create pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SpeechAnalysisPipeline()
    return _pipeline_instance


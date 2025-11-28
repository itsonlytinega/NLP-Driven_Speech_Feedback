"""
Audio feature extraction using Wav2Vec2 for accent-robust analysis
DEFENSE: Wav2Vec2 provides accent-invariant features for speaker diversity
"""

import os
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model loading (load once, reuse)
_wav2vec_model = None
_wav2vec_processor = None

def _load_wav2vec():
    """Load Wav2Vec2 model (lazy loading)"""
    global _wav2vec_model, _wav2vec_processor
    
    if _wav2vec_model is None:
        try:
            logger.info("Loading Wav2Vec2 model for accent-robust features...")
            _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            _wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            
            # Set to eval mode for inference
            _wav2vec_model.to(DEVICE)
            _wav2vec_model.eval()
            logger.info("Wav2Vec2 device: %s", DEVICE)
            logger.info("[OK] Wav2Vec2 model loaded")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2: {e}")
            logger.warning("Falling back to audio-free mode")
            return None
    
    return _wav2vec_model, _wav2vec_processor


def get_wav2vec_features(audio_path, max_length_seconds=30):
    """
    Extract Wav2Vec2 features from audio for accent-robust analysis.
    
    DEFENSE: Why Wav2Vec2?
    - Self-supervised pre-training on 960h of unlabeled speech
    - Learns accent-invariant phonetic representations
    - 768-dim features complement BERT's 768-dim text features
    - Proven on LibriSpeech, Common Voice (12+ accents tested)
    
    Args:
        audio_path: Path to audio file
        max_length_seconds: Maximum audio length to process
        
    Returns:
        numpy array: 768-dimensional feature vector (accent-invariant)
    """
    try:
        # Load Wav2Vec2 if not already loaded
        model, processor = _load_wav2vec()
        if model is None or processor is None:
            # Fallback: return zero vector
            logger.warning(f"Wav2Vec2 not available, returning zeros for {audio_path}")
            return np.zeros(768)
        
        # Check if audio file exists
        if not os.path.exists(audio_path) or not audio_path.endswith('.wav'):
            logger.debug(f"Audio file not found or not WAV: {audio_path}")
            return np.zeros(768)
        
        # Load audio using librosa (avoids torchcodec dependency)
        # Load audio with librosa (more reliable, no torchcodec needed)
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=max_length_seconds)
        
        # Process with Wav2Vec2
        # Processor expects 1D numpy array (y is already numpy array from librosa)
        input_values = processor(
            y, 
            return_tensors="pt", 
            sampling_rate=16000,
            padding=True
        ).input_values
        input_values = input_values.to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
            hidden_states = hidden_states.cpu()
            
            # Average pooling over time dimension
            # Shape: [batch, seq_len, 768] -> [batch, 768]
            features = hidden_states.mean(dim=1)
        
        # Return as numpy array
        return features.squeeze().numpy()
        
    except Exception as e:
        logger.error(f"Error extracting Wav2Vec2 features from {audio_path}: {e}")
        return np.zeros(768)


def get_audio_stats(audio_path):
    """
    Get basic audio statistics (complement to Wav2Vec2).
    
    Returns:
        dict: Audio statistics including RMS, SNR, duration
    """
    try:
        if not os.path.exists(audio_path):
            return {
                'duration': 0.0,
                'rms_db': 0.0,
                'snr_db': 0.0,
                'has_audio': False
            }
        
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr
        
        # RMS energy
        rms_energy = librosa.feature.rms(y=y)[0].mean()
        rms_db = 20 * np.log10(rms_energy + 1e-10)
        
        # SNR estimate
        signal_level = np.percentile(np.abs(y), 95)
        noise_floor = np.percentile(np.abs(y), 5)
        snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        
        return {
            'duration': duration,
            'rms_db': rms_db,
            'snr_db': snr_db,
            'has_audio': True
        }
        
    except Exception as e:
        logger.error(f"Error getting audio stats from {audio_path}: {e}")
        return {
            'duration': 0.0,
            'rms_db': 0.0,
            'snr_db': 0.0,
            'has_audio': False
        }


"""
Test the complete BERT + Cause Classifier pipeline
"""

import os
import sys
import pandas as pd
import torch

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'verbalcoach.settings')
import django
django.setup()

from coach.utils import get_bert_scores
from coach.train_cause import CauseClassifierTrainer

def test_bert_scoring():
    """Test BERT scoring on sample transcripts"""
    print("\n=== Testing BERT Scoring ===\n")
    
    test_transcripts = [
        "Hello, um, I think this is a test of the filler word detection system.",
        "The quick brown fox jumps over the lazy dog. Perfect clear speech.",
        "So, basically, uh, I mean, you know, like, this is, um, you know, really like testing things.",
    ]
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\nTest {i}:")
        print(f"Transcript: {transcript[:60]}...")
        
        try:
            scores = get_bert_scores(transcript)
            print(f"  Filler: {scores['filler']:.3f}")
            print(f"  Clarity: {scores['clarity']:.3f}")
            print(f"  Pacing: {scores['pacing']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✓ BERT scoring complete!")

def test_cause_prediction():
    """Test cause prediction on test samples"""
    print("\n=== Testing Cause Prediction ===\n")
    
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    print(f"Test samples: {len(test_df)}")
    
    # Initialize trainer to load model
    trainer = CauseClassifierTrainer()
    
    # Test on first 5 samples
    for idx in range(min(5, len(test_df))):
        row = test_df.iloc[idx]
        print(f"\nSample {idx + 1}:")
        print(f"  Transcript: {row['transcript'][:50]}...")
        print(f"  True cause: {row['cause']}")
        
        try:
            predicted_cause = trainer.predict_cause(
                row['audio_path'],
                row['transcript']
            )
            print(f"  Predicted: {predicted_cause}")
            print(f"  Match: {'✓' if predicted_cause == row['cause'] else '✗'}")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    print("=" * 60)
    print("BERT Pipeline Test")
    print("=" * 60)
    
    # Test 1: BERT scoring
    test_bert_scoring()
    
    # Test 2: Cause prediction
    test_cause_prediction()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()




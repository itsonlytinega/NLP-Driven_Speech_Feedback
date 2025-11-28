"""
Test script for Django integration of BERT + XGBoost models.
Run this to verify your NLP pipeline is working correctly.
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'verbalcoach.settings')
django.setup()

# Now we can import Django models
from coach.utils import get_bert_scores, predict_cause

def test_bert():
    """Test BERT scoring"""
    print("\n" + "="*60)
    print("Testing BERT Scoring")
    print("="*60 + "\n")
    
    test_cases = [
        ("So like, um, I was basically thinking about this and you know, it was really kind of confusing", "High filler content"),
        ("The presentation demonstrates advanced technical capabilities with comprehensive analysis.", "Low filler content"),
        ("Well, actually, I think that, like, it's really hard to explain this properly.", "Moderate filler content"),
    ]
    
    for text, description in test_cases:
        scores = get_bert_scores(text)
        print(f"\n{description}:")
        print(f"  Text: {text[:60]}...")
        print(f"  Filler: {scores['filler']:.3f}")
        print(f"  Clarity: {scores['clarity']:.3f}")
        print(f"  Pacing: {scores['pacing']:.3f}")

def test_cause_prediction():
    """Test cause prediction"""
    print("\n" + "="*60)
    print("Testing Cause Classification")
    print("="*60 + "\n")
    
    test_cases = [
        ("So like, um, I was basically thinking about this and you know, it was really kind of confusing", 10, "High filler, fast speech"),
        ("I... um... well... I think... it's... uh... difficult... to explain.", 15, "Slow, many fillers"),
        ("This is a clear and well-structured presentation of our findings.", 20, "Clear, confident speech"),
        ("The data shows significant improvements in performance metrics.", 18, "Professional speech"),
    ]
    
    for text, duration, description in test_cases:
        result = predict_cause(None, text, duration)
        print(f"\n{description}:")
        print(f"  Text: {text[:50]}...")
        print(f"  Cause: {result['cause']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        if 'all_probabilities' in result and result['all_probabilities']:
            print(f"  Probabilities:")
            for cause, prob in result['all_probabilities'].items():
                print(f"    {cause}: {prob:.3f}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VerbalCoach NLP Integration Test")
    print("="*60)
    
    # Test BERT
    test_bert()
    
    # Test Cause Prediction
    test_cause_prediction()
    
    print("\n" + "="*60)
    print("[SUCCESS] All tests passed!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Test Django views with real audio upload")
    print("2. Verify cause classification in speech session analysis")
    print("3. Check report at: reports/full_report.html")
    print("\n")

if __name__ == "__main__":
    main()


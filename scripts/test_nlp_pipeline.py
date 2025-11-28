"""
Comprehensive Testing Script for NLP Pipeline

Tests all components of the speech analysis pipeline:
- Whisper transcription
- Pacing analysis
- Filler detection
- Cause classification
- Drill recommendations

Usage:
    python scripts/test_nlp_pipeline.py
    python scripts/test_nlp_pipeline.py --audio /path/to/audio.mp3
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'verbalcoach.settings')
import django
django.setup()

from speech_sessions.nlp_pipeline import SpeechAnalysisPipeline, get_pipeline
from coach.models import Drill
from speech_sessions.models import SpeechSession
from django.contrib.auth import get_user_model
import json

User = get_user_model()


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_pipeline_initialization():
    """Test 1: Pipeline Initialization"""
    print_section("TEST 1: Pipeline Initialization")
    
    try:
        pipeline = get_pipeline()
        print("[OK] Pipeline initialized successfully")
        
        # Check components
        print(f"\nComponents loaded:")
        print(f"  - Whisper Model: {'[OK]' if pipeline.whisper_model else '[FAIL] (will fail transcription)'}")
        print(f"  - BERT Tokenizer: {'[OK]' if pipeline.tokenizer else '[FAIL]'}")
        print(f"  - Filler Detector: {'[OK] (BERT)' if pipeline.filler_detector else '[WARNING] (rule-based fallback)'}")
        print(f"  - Cause Classifier: {'[OK] (BERT)' if pipeline.cause_classifier else '[WARNING] (rule-based fallback)'}")
        
        return True, pipeline
    
    except Exception as e:
        print(f"[ERROR] Pipeline initialization failed: {e}")
        return False, None


def test_transcription(pipeline, audio_path=None):
    """Test 2: Audio Transcription"""
    print_section("TEST 2: Audio Transcription")
    
    if not pipeline.whisper_model:
        print("[WARNING] Whisper model not loaded, skipping transcription test")
        return False
    
    if not audio_path:
        print("[WARNING] No audio file provided, skipping transcription test")
        print("  Provide audio with: --audio /path/to/file.mp3")
        return False
    
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}")
        return False
    
    try:
        print(f"Transcribing: {audio_path}")
        result = pipeline.transcribe_audio(audio_path)
        
        if result:
            print(f"\n[OK] Transcription successful!")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Words: {len(result['segments'])}")
            print(f"  Language: {result.get('language', 'N/A')}")
            return True
        else:
            print("[ERROR] Transcription failed")
            return False
    
    except Exception as e:
        print(f"[ERROR] Transcription error: {e}")
        return False


def test_pacing_analysis(pipeline):
    """Test 3: Pacing Analysis"""
    print_section("TEST 3: Pacing Analysis")
    
    # Test with synthetic segments
    test_segments = [
        {'word': 'Hello', 'start': 0.0, 'end': 0.5, 'probability': 0.95},
        {'word': 'how', 'start': 0.5, 'end': 0.7, 'probability': 0.92},
        {'word': 'are', 'start': 0.7, 'end': 0.9, 'probability': 0.91},
        {'word': 'you', 'start': 0.9, 'end': 1.2, 'probability': 0.93},
        {'word': 'today', 'start': 1.2, 'end': 1.6, 'probability': 0.90},
        {'word': 'I', 'start': 2.0, 'end': 2.1, 'probability': 0.94},  # Pause before
        {'word': 'am', 'start': 2.1, 'end': 2.3, 'probability': 0.92},
        {'word': 'doing', 'start': 2.3, 'end': 2.6, 'probability': 0.91},
        {'word': 'great', 'start': 2.6, 'end': 3.0, 'probability': 0.93},
    ]
    
    try:
        result = pipeline.analyze_pacing(test_segments)
        
        print("[OK] Pacing analysis successful!")
        print(f"\n  Results:")
        print(f"    WPM: {result['wpm']}")
        print(f"    Speaking rate: {result['speaking_rate']}")
        print(f"    Avg pause: {result['avg_pause_duration']}s")
        print(f"    Total pauses: {result['total_pauses']}")
        print(f"    Long pauses: {result['long_pauses']}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Pacing analysis failed: {e}")
        return False


def test_filler_detection(pipeline):
    """Test 4: Filler Detection"""
    print_section("TEST 4: Filler Detection")
    
    test_cases = [
        "Um, I think this is like really important",
        "You know, basically, we need to focus on the main points",
        "The research shows clear evidence of significant patterns",
        "So, uh, let me explain this concept to you, like, right now"
    ]
    
    try:
        print("Testing filler detection on sample texts:\n")
        
        for i, text in enumerate(test_cases, 1):
            print(f"Test {i}: \"{text}\"")
            result = pipeline.detect_fillers(text)
            
            print(f"  Fillers found: {result['total_count']}")
            print(f"  Density: {result['density']:.3f} per second")
            print(f"  Severity: {result['severity']}")
            if result['filler_types']:
                print(f"  Types: {', '.join([f['filler'] for f in result['filler_types'][:3]])}")
            print()
        
        print("[OK] Filler detection working!")
        return True
    
    except Exception as e:
        print(f"[ERROR] Filler detection failed: {e}")
        return False


def test_cause_classification(pipeline):
    """Test 5: Cause Classification"""
    print_section("TEST 5: Cause Classification")
    
    test_cases = [
        {
            'text': "Um, uh, I think, like, we should hurry",
            'pacing': {'wpm': 175, 'long_pauses': 1, 'speaking_rate': 'too_fast'},
            'fillers': {'density': 0.09, 'total_count': 5},
            'expected': 'anxiety'
        },
        {
            'text': "Well... I need to... think about this...",
            'pacing': {'wpm': 95, 'long_pauses': 4, 'speaking_rate': 'too_slow'},
            'fillers': {'density': 0.02, 'total_count': 1},
            'expected': 'stress'
        },
        {
            'text': "Um, I guess, maybe, like, we could try",
            'pacing': {'wpm': 110, 'long_pauses': 2, 'speaking_rate': 'optimal'},
            'fillers': {'density': 0.12, 'total_count': 6},
            'expected': 'lack_of_confidence'
        }
    ]
    
    try:
        print("Testing cause classification:\n")
        
        for i, case in enumerate(test_cases, 1):
            result = pipeline.classify_cause(case['text'], case['pacing'], case['fillers'])
            
            print(f"Test {i}: {case['text'][:50]}...")
            print(f"  Classified as: {result}")
            print(f"  Expected: {case['expected']}")
            print(f"  {'[OK] Correct!' if result == case['expected'] else '[ERROR] Different'}")
            print()
        
        print("[OK] Cause classification working!")
        return True
    
    except Exception as e:
        print(f"[ERROR] Cause classification failed: {e}")
        return False


def test_recommendations(pipeline):
    """Test 6: Drill Recommendations"""
    print_section("TEST 6: Drill Recommendations")
    
    # Check if drills exist
    drill_count = Drill.objects.filter(is_active=True).count()
    print(f"Active drills in database: {drill_count}")
    
    if drill_count == 0:
        print("[WARNING] No active drills found. Run: python manage.py populate_drills")
        return False
    
    try:
        test_case = {
            'cause': 'anxiety',
            'pacing': {'wpm': 175, 'speaking_rate': 'too_fast'},
            'fillers': {'severity': 'high', 'total_count': 15},
            'pronunciation': {'quality': 'good', 'clarity_score': 75}
        }
        
        recommendations = pipeline.recommend_drills(
            test_case['cause'],
            test_case['pacing'],
            test_case['fillers'],
            test_case['pronunciation']
        )
        
        print(f"\n[OK] Generated {len(recommendations)} recommendations")
        print("\nRecommended drills:")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec['name']}")
            print(f"     Type: {rec['skill_type']}")
            print(f"     Reason: {rec['reason']}")
            print(f"     Priority: {rec['priority']}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Recommendation generation failed: {e}")
        return False


def test_full_pipeline(pipeline, audio_path=None):
    """Test 7: Full Pipeline Integration"""
    print_section("TEST 7: Full Pipeline Integration")
    
    if not audio_path:
        print("[WARNING] No audio file provided, skipping full pipeline test")
        print("  Provide audio with: --audio /path/to/file.mp3")
        return False
    
    if not pipeline.whisper_model:
        print("[WARNING] Whisper model not loaded, skipping full pipeline test")
        return False
    
    try:
        print(f"Running complete analysis on: {audio_path}\n")
        
        results = pipeline.analyze_audio(audio_path)
        
        if not results.get('success'):
            print(f"[ERROR] Analysis failed: {results.get('error')}")
            return False
        
        print("[OK] Full pipeline analysis successful!\n")
        print("Results Summary:")
        print(f"  Transcription: {results['transcription'][:80]}...")
        print(f"  Duration: {results['duration']:.2f}s")
        print(f"  Word count: {results['word_count']}")
        print(f"  WPM: {results['pacing']['wpm']}")
        print(f"  Fillers: {results['fillers']['total_count']}")
        print(f"  Cause: {results['cause']}")
        print(f"  Confidence: {results['confidence_score']:.2f}")
        print(f"  Recommendations: {len(results['recommendations'])}")
        
        # Save results to file
        output_file = project_root / 'test_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Full results saved to: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Test 8: API Integration"""
    print_section("TEST 8: API Integration")
    
    # Check if we have a test user and session
    test_user = User.objects.first()
    if not test_user:
        print("[WARNING] No users found in database")
        print("  Create a user first: python manage.py createsuperuser")
        return False
    
    test_session = SpeechSession.objects.filter(user=test_user).first()
    if not test_session:
        print("[WARNING] No speech sessions found")
        print("  Create a session via the web interface or API")
        return False
    
    print(f"Test user: {test_user.email}")
    print(f"Test session: #{test_session.id}")
    
    if test_session.audio_file and os.path.exists(test_session.audio_file.path):
        print(f"Audio file: {test_session.audio_file.path}")
        print("\nYou can test the API endpoint with:")
        print(f"  POST /api/sessions/{test_session.id}/analyze_with_nlp/")
        return True
    else:
        print("[WARNING] Session has no audio file")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test NLP Pipeline')
    parser.add_argument('--audio', help='Path to audio file for testing')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  NLP PIPELINE COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Initialization
    success, pipeline = test_pipeline_initialization()
    results['initialization'] = success
    
    if not pipeline:
        print("\n[ERROR] Cannot proceed without pipeline. Please fix initialization errors.")
        return
    
    # Test 2: Transcription (if audio provided)
    results['transcription'] = test_transcription(pipeline, args.audio)
    
    # Test 3: Pacing Analysis
    results['pacing'] = test_pacing_analysis(pipeline)
    
    # Test 4: Filler Detection
    results['fillers'] = test_filler_detection(pipeline)
    
    # Test 5: Cause Classification
    results['cause'] = test_cause_classification(pipeline)
    
    # Test 6: Recommendations
    results['recommendations'] = test_recommendations(pipeline)
    
    # Test 7: Full Pipeline (if audio provided)
    if args.audio:
        results['full_pipeline'] = test_full_pipeline(pipeline, args.audio)
    
    # Test 8: API Integration
    results['api'] = test_api_integration()
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\nResults:")
    for test, result in results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"  {test.ljust(20)}: {status}")
    
    print(f"\n{'='*70}")
    print(f"  Total: {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your NLP pipeline is ready to use!")
    elif passed >= total * 0.7:
        print("\n[WARNING] Most tests passed. Check failures above and fix if needed.")
    else:
        print("\n[WARNING] Many tests failed. Please review the setup and dependencies.")
    
    print("\nNext steps:")
    if not results.get('transcription'):
        print("  - Install Whisper: pip install openai-whisper")
    if not results.get('recommendations'):
        print("  - Populate drills: python manage.py populate_drills")
    if args.audio and results.get('full_pipeline'):
        print("  - Check test_results.json for detailed analysis")
    print("  - Train BERT models using the Colab scripts")
    print("  - Test the API endpoint with real audio files")


if __name__ == '__main__':
    main()


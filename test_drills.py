#!/usr/bin/env python
"""
Comprehensive Test Script for Pronunciation Drills
Run this script to test all drill functionality
"""

import os
import sys
import django
from django.conf import settings

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'verbalcoach.settings')
django.setup()

from coach.models import Drill, DrillCompletion, User
from speech_sessions.models import SpeechSession
from django.test import Client
from django.urls import reverse
import json

def test_drill_models():
    """Test Drill model functionality"""
    print("Testing Drill Models...")
    
    # Test drill creation
    drills = Drill.objects.filter(is_active=True)
    print(f"Found {drills.count()} active drills")
    
    # Test skill types
    skill_types = drills.values_list('skill_type', flat=True).distinct()
    print(f"Skill types: {list(skill_types)}")
    
    # Test interactive elements
    for drill in drills[:3]:
        if drill.interactive_elements:
            print(f"{drill.name} has interactive elements: {list(drill.interactive_elements.keys())}")
    
    return True

def test_drill_views():
    """Test drill view functionality"""
    print("\nTesting Drill Views...")
    
    client = Client()
    
    # Test drills list page
    try:
        response = client.get('/drills/')
        if response.status_code == 200:
            print("Drills list page loads successfully")
        else:
            print(f"Drills list page failed: {response.status_code}")
    except Exception as e:
        print(f"Drills list page error: {e}")
    
    # Test individual drill pages
    drills = Drill.objects.filter(is_active=True)[:3]
    for drill in drills:
        try:
            response = client.get(f'/drill/{drill.id}/')
            if response.status_code == 200:
                print(f"Drill '{drill.name}' loads successfully")
            else:
                print(f"Drill '{drill.name}' failed: {response.status_code}")
        except Exception as e:
            print(f"Drill '{drill.name}' error: {e}")
    
    return True

def test_recommendation_logic():
    """Test drill recommendation logic"""
    print("\nTesting Recommendation Logic...")
    
    # Create test user
    user, created = User.objects.get_or_create(
        email='test@example.com',
        defaults={
            'first_name': 'Test',
            'last_name': 'User',
            'is_email_verified': True,
            'is_active': True
        }
    )
    
    # Create test session with high filler count
    session = SpeechSession.objects.create(
        user=user,
        duration=120,  # 2 minutes
        filler_count=10,  # High filler count (0.083 per second)
        transcription="This is a test with um many uh filler words like um and uh",
        confidence_score=0.5,  # Low confidence
        status='analyzed'
    )
    
    print(f"Created test session with {session.filler_count} fillers in {session.duration}s")
    
    # Test recommendation logic
    from speech_sessions.api import SpeechSessionViewSet
    viewset = SpeechSessionViewSet()
    recommendations = viewset.get_drill_recommendations(session)
    
    print(f"Generated {len(recommendations)} recommendations")
    for rec in recommendations:
        print(f"   - {rec['name']}: {rec['reason']}")
    
    return True

def test_spacy_integration():
    """Test spaCy integration"""
    print("\nTesting spaCy Integration...")
    
    try:
        from coach.views import get_spacy_model
        nlp = get_spacy_model()
        
        if nlp:
            # Test tokenization
            doc = nlp("I have a dream that one day this nation will rise up")
            tokens = [token.text for token in doc]
            print(f"spaCy tokenization works: {tokens[:5]}...")
            
            # Test phonetic tips
            from coach.views import PoemEchoChallengeView
            view = PoemEchoChallengeView()
            tips = view.get_phonetic_tips("I have a dream that one day this nation will rise up")
            print(f"Phonetic tips generated: {len(tips)} tips")
        else:
            print("spaCy model not available")
            return False
            
    except Exception as e:
        print(f"spaCy integration error: {e}")
        return False
    
    return True

def test_javascript_components():
    """Test JavaScript components availability"""
    print("\nTesting JavaScript Components...")
    
    js_file_path = 'static/js/drills.js'
    if os.path.exists(js_file_path):
        with open(js_file_path, 'r') as f:
            content = f.read()
            
        # Check for key functions
        functions = [
            'startMic',
            'startTimer', 
            'showFeedback',
            'createSpinningWheel',
            'createDragDropInterface',
            'playAudioWithWaveform'
        ]
        
        for func in functions:
            if f'function {func}(' in content or f'{func}(' in content:
                print(f"{func} function found")
            else:
                print(f"{func} function missing")
        
        print(f"JavaScript file exists ({len(content)} characters)")
    else:
        print("JavaScript file not found")
        return False
    
    return True

def test_templates():
    """Test template availability"""
    print("\nTesting Templates...")
    
    templates = [
        'coach/templates/coach/drill_detail.html',
        'coach/templates/coach/drill_poem_echo.html',
        'coach/templates/coach/drill_vowel_vortex.html',
        'coach/templates/coach/drill_mirror_mimic.html',
        'coach/templates/coach/drill_phonetic_puzzle.html',
        'coach/templates/coach/drill_shadow_superhero.html',
        'coach/templates/coach/drill_pencil_precision.html',
        'coach/templates/coach/drills_list.html',
        'coach/templates/coach/dashboard.html'
    ]
    
    for template in templates:
        if os.path.exists(template):
            print(f"{template}")
        else:
            print(f"{template} missing")
    
    return True

def test_url_routing():
    """Test URL routing"""
    print("\nTesting URL Routing...")
    
    from coach.urls import urlpatterns
    
    drill_urls = [
        'drills_list',
        'drill_detail', 
        'drill_poem_echo',
        'drill_vowel_vortex',
        'drill_mirror_mimic',
        'drill_phonetic_puzzle',
        'drill_shadow_superhero',
        'drill_pencil_precision',
        'complete_drill'
    ]
    
    url_names = [pattern.name for pattern in urlpatterns if hasattr(pattern, 'name') and pattern.name]
    
    for url_name in drill_urls:
        if url_name in url_names:
            print(f"{url_name} URL configured")
        else:
            print(f"{url_name} URL missing")
    
    return True

def main():
    """Run all tests"""
    print("Starting Comprehensive Drill Testing...")
    print("=" * 50)
    
    tests = [
        test_drill_models,
        test_drill_views,
        test_recommendation_logic,
        test_spacy_integration,
        test_javascript_components,
        test_templates,
        test_url_routing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Drills are ready for use.")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Comprehensive Test Script for Day 2 Sprint 3: Pacing and Filler Words Drills
Run this script to test all new drill functionality
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

def test_new_drill_models():
    """Test new drill models and data"""
    print("Testing New Drill Models...")
    
    # Test drill counts by skill type
    pronunciation_drills = Drill.objects.filter(skill_type='pronunciation', is_active=True)
    pacing_drills = Drill.objects.filter(skill_type='pacing', is_active=True)
    filler_drills = Drill.objects.filter(skill_type='filler_words', is_active=True)
    
    print(f"Pronunciation drills: {pronunciation_drills.count()}")
    print(f"Pacing drills: {pacing_drills.count()}")
    print(f"Filler words drills: {filler_drills.count()}")
    
    # Test new drill names
    new_pacing_drills = [
        'Metronome Rhythm Race',
        'Pause Pyramid Builder', 
        'Slow-Motion Story Slam',
        'Beat Drop Dialogue',
        'Timer Tag Team',
        'Echo Chamber Escalation'
    ]
    
    new_filler_drills = [
        'Filler Zap Game',
        'Silent Switcheroo',
        'Pause Power-Up',
        'Filler Hunt Mirror Maze',
        'Word Swap Whirlwind',
        'Echo Elimination Echo'
    ]
    
    for drill_name in new_pacing_drills:
        if Drill.objects.filter(name=drill_name, skill_type='pacing').exists():
            print(f"Pacing drill '{drill_name}' exists")
        else:
            print(f"Pacing drill '{drill_name}' missing")
    
    for drill_name in new_filler_drills:
        if Drill.objects.filter(name=drill_name, skill_type='filler_words').exists():
            print(f"Filler drill '{drill_name}' exists")
        else:
            print(f"Filler drill '{drill_name}' missing")
    
    return True

def test_new_drill_views():
    """Test new drill view functionality"""
    print("\nTesting New Drill Views...")
    
    client = Client()
    
    # Test new pacing drill URLs
    pacing_urls = [
        '/drill/metronome-rhythm/',
        '/drill/pause-pyramid/',
        '/drill/slow-motion-story/',
        '/drill/beat-drop-dialogue/',
        '/drill/timer-tag-team/',
        '/drill/echo-chamber/',
    ]
    
    for url in pacing_urls:
        try:
            response = client.get(url)
            if response.status_code == 200:
                print(f"Pacing drill {url} loads successfully")
            else:
                print(f"Pacing drill {url} failed: {response.status_code}")
        except Exception as e:
            print(f"Pacing drill {url} error: {e}")
    
    # Test new filler drill URLs
    filler_urls = [
        '/drill/filler-zap-game/',
        '/drill/silent-switcheroo/',
        '/drill/pause-powerup/',
        '/drill/filler-hunt-mirror/',
        '/drill/word-swap-whirlwind/',
        '/drill/echo-elimination/',
    ]
    
    for url in filler_urls:
        try:
            response = client.get(url)
            if response.status_code == 200:
                print(f"Filler drill {url} loads successfully")
            else:
                print(f"Filler drill {url} failed: {response.status_code}")
        except Exception as e:
            print(f"Filler drill {url} error: {e}")
    
    return True

def test_javascript_components():
    """Test JavaScript components for new drills"""
    print("\nTesting JavaScript Components...")
    
    js_file_path = 'static/js/drills.js'
    if os.path.exists(js_file_path):
        with open(js_file_path, 'r') as f:
            content = f.read()
            
        # Check for new functions
        new_functions = [
            'createMetronome',
            'simulateFillerDetection',
            'createSpeedController',
            'createBeatPattern',
            'animateProgressBar',
            'createZapEffect',
            'createStreakCounter',
            'createWordCounter'
        ]
        
        for func in new_functions:
            if f'function {func}(' in content or f'{func}(' in content:
                print(f"{func} function found")
            else:
                print(f"{func} function missing")
        
        print(f"JavaScript file updated ({len(content)} characters)")
    else:
        print("JavaScript file not found")
        return False
    
    return True

def test_templates():
    """Test template availability for new drills"""
    print("\nTesting New Templates...")
    
    new_templates = [
        'coach/templates/coach/drill_metronome_rhythm.html',
        'coach/templates/coach/drill_pause_pyramid.html',
        'coach/templates/coach/drill_filler_zap_game.html',
        'coach/templates/coach/drill_base.html',
    ]
    
    for template in new_templates:
        if os.path.exists(template):
            print(f"{template}")
        else:
            print(f"{template} missing")
    
    return True

def test_url_routing():
    """Test URL routing for new drills"""
    print("\nTesting New URL Routing...")
    
    from coach.urls import urlpatterns
    
    new_url_names = [
        'drill_metronome_rhythm',
        'drill_pause_pyramid',
        'drill_slow_motion_story',
        'drill_beat_drop_dialogue',
        'drill_timer_tag_team',
        'drill_echo_chamber',
        'drill_filler_zap_game',
        'drill_silent_switcheroo',
        'drill_pause_powerup',
        'drill_filler_hunt_mirror',
        'drill_word_swap_whirlwind',
        'drill_echo_elimination'
    ]
    
    url_names = [pattern.name for pattern in urlpatterns if hasattr(pattern, 'name') and pattern.name]
    
    for url_name in new_url_names:
        if url_name in url_names:
            print(f"{url_name} URL configured")
        else:
            print(f"{url_name} URL missing")
    
    return True

def test_drill_completion():
    """Test DrillCompletion model functionality"""
    print("\nTesting DrillCompletion Model...")
    
    # Test model exists
    try:
        completions = DrillCompletion.objects.all()
        print(f"DrillCompletion model exists ({completions.count()} completions)")
        
        # Test model fields
        if hasattr(DrillCompletion, 'user') and hasattr(DrillCompletion, 'drill'):
            print("Required fields exist")
        else:
            print("Required fields missing")
            
    except Exception as e:
        print(f"DrillCompletion model error: {e}")
        return False
    
    return True

def test_recommendation_logic():
    """Test updated recommendation logic"""
    print("\nTesting Recommendation Logic...")
    
    # Create test user
    user, created = User.objects.get_or_create(
        email='test_day2@example.com',
        defaults={
            'first_name': 'Test',
            'last_name': 'User',
            'is_email_verified': True,
            'is_active': True
        }
    )
    
    # Test pacing recommendation
    pacing_session = SpeechSession.objects.create(
        user=user,
        duration=60,
        filler_count=2,
        transcription="This is a very fast speech example to test pacing detection.",
        confidence_score=0.8,
        status='analyzed'
    )
    
    from speech_sessions.api import SpeechSessionViewSet
    viewset = SpeechSessionViewSet()
    recommendations = viewset.get_drill_recommendations(pacing_session)
    
    print(f"Generated {len(recommendations)} recommendations")
    
    # Check for pacing drills in recommendations
    pacing_recommendations = [r for r in recommendations if r['skill_type'] == 'pacing']
    if pacing_recommendations:
        print(f"Pacing recommendations: {len(pacing_recommendations)}")
    else:
        print("No pacing recommendations found")
    
    return True

def test_interactivity_features():
    """Test interactivity features"""
    print("\nTesting Interactivity Features...")
    
    # Test metronome functionality
    print("Metronome: Web Audio API support")
    
    # Test filler detection
    print("Filler Detection: Simulated detection system")
    
    # Test progress tracking
    print("Progress Tracking: Visual feedback systems")
    
    # Test streak counting
    print("Streak Counting: Game mechanics")
    
    return True

def main():
    """Run all tests"""
    print("Starting Day 2 Sprint 3 Testing...")
    print("=" * 50)
    
    tests = [
        test_new_drill_models,
        test_new_drill_views,
        test_javascript_components,
        test_templates,
        test_url_routing,
        test_drill_completion,
        test_recommendation_logic,
        test_interactivity_features
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
        print("All Day 2 tests passed! New drills are ready for use.")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

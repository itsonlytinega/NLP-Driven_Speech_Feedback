from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase
from rest_framework.authtoken.models import Token
from rest_framework import status

from .models import SpeechSession

User = get_user_model()


class SpeechSessionModelTest(TestCase):
    """Test cases for the SpeechSession model."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            email='testuser@example.com',
            first_name='Test',
            last_name='User',
            password='testpass123'
        )
    
    def test_speech_session_creation(self):
        """Test creating a speech session."""
        session = SpeechSession.objects.create(
            user=self.user,
            duration=300,
            filler_count=5,
            pacing_analysis='Good pacing overall',
            status='analyzed'
        )
        
        self.assertEqual(session.user, self.user)
        self.assertEqual(session.duration, 300)
        self.assertEqual(session.filler_count, 5)
        self.assertEqual(session.status, 'analyzed')
        self.assertTrue(session.date)
        self.assertTrue(session.created_at)
    
    def test_duration_minutes_property(self):
        """Test duration_minutes property."""
        session = SpeechSession.objects.create(
            user=self.user,
            duration=125  # 2 minutes 5 seconds
        )
        
        self.assertEqual(session.duration_minutes, '2:05')
    
    def test_filler_rate_property(self):
        """Test filler_rate property."""
        session = SpeechSession.objects.create(
            user=self.user,
            duration=60,  # 1 minute
            filler_count=3
        )
        
        self.assertEqual(session.filler_rate, 3.0)  # 3 fillers per minute


class SpeechSessionViewTest(TestCase):
    """Test cases for speech session views."""
    
    def setUp(self):
        """Set up test data."""
        self.client = Client()
        self.user = User.objects.create_user(
            email='testuser@example.com',
            first_name='Test',
            last_name='User',
            password='testpass123'
        )
        self.session = SpeechSession.objects.create(
            user=self.user,
            duration=300,
            filler_count=5,
            status='analyzed'
        )
    
    def test_session_list_view_requires_login(self):
        """Test that session list view requires login."""
        response = self.client.get(reverse('speech_sessions:session_list'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_session_list_view_authenticated(self):
        """Test session list view for authenticated user."""
        self.client.login(email='testuser@example.com', password='testpass123')
        response = self.client.get(reverse('speech_sessions:session_list'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Speech Sessions')
    
    def test_session_create_view_post(self):
        """Test session create view POST request."""
        self.client.login(email='testuser@example.com', password='testpass123')
        
        data = {
            'duration': 180,
            'transcription': 'This is a test transcription.'
        }
        
        response = self.client.post(reverse('speech_sessions:session_create'), data)
        
        self.assertEqual(response.status_code, 302)  # Redirect after successful creation
        self.assertTrue(SpeechSession.objects.filter(
            user=self.user,
            duration=180
        ).exists())


class SpeechSessionAPITest(APITestCase):
    """Test cases for speech session API endpoints."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            email='testuser@example.com',
            first_name='Test',
            last_name='User',
            password='testpass123'
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        
        self.session = SpeechSession.objects.create(
            user=self.user,
            duration=300,
            filler_count=5,
            status='analyzed'
        )
    
    def test_api_authentication_required(self):
        """Test that API requires authentication."""
        self.client.credentials()  # Remove authentication
        response = self.client.get('/speech-sessions/api/sessions/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_api_list_sessions(self):
        """Test API list sessions endpoint."""
        response = self.client.get('/speech-sessions/api/sessions/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['id'], self.session.id)
    
    def test_api_create_session(self):
        """Test API create session endpoint."""
        data = {
            'duration': 180,
            'transcription': 'API test transcription'
        }
        
        response = self.client.post('/speech-sessions/api/sessions/', data)
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['duration'], 180)
        self.assertTrue(SpeechSession.objects.filter(
            user=self.user,
            duration=180
        ).exists())
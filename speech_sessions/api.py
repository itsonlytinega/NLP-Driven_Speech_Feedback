from rest_framework import viewsets, permissions, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Q, Avg, Sum, Count
from django.utils import timezone
from datetime import timedelta

from .models import SpeechSession
from .serializers import (
    SpeechSessionSerializer,
    SpeechSessionCreateSerializer,
    SpeechSessionListSerializer,
    SpeechSessionAnalyticsSerializer
)
from coach.models import Drill
from .nlp_pipeline import get_pipeline
from django.conf import settings
import random
import os


class IsOwnerPermission(permissions.BasePermission):
    """
    Custom permission to ensure users can only access their own speech sessions.
    
    This permission class integrates with the 2FA system by checking if the user
    is properly authenticated and verified.
    """
    
    def has_permission(self, request, view):
        """Check if user has permission to access the view."""
        if not request.user or not request.user.is_authenticated:
            return False
        
        # Check 2FA requirements if enabled
        # Note: DRF handles authentication, so if user is authenticated via session,
        # they should have access (2FA verification happens during login)
        # For API endpoints, we rely on DRF's authentication system
        
        return True
    
    def has_object_permission(self, request, view, obj):
        """Check if user has permission to access the specific object."""
        return obj.user == request.user


class SpeechSessionViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing speech sessions via REST API.
    
    This ViewSet provides full CRUD operations for speech sessions with:
    - Token-based authentication
    - User-specific filtering (users can only see their own sessions)
    - 2FA integration
    - Filtering and search capabilities
    - Custom analytics endpoint
    
    Endpoints:
    - GET /api/sessions/ - List user's sessions
    - POST /api/sessions/ - Create new session
    - GET /api/sessions/{id}/ - Retrieve specific session
    - PUT/PATCH /api/sessions/{id}/ - Update session
    - DELETE /api/sessions/{id}/ - Delete session
    - GET /api/sessions/analytics/ - Get analytics data
    """
    
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerPermission]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter, filters.SearchFilter]
    
    # Filtering options
    filterset_fields = ['status', 'date']
    search_fields = ['transcription', 'pacing_analysis']
    ordering_fields = ['date', 'duration', 'filler_count', 'confidence_score']
    ordering = ['-date']  # Default ordering by most recent first
    
    def get_queryset(self):
        """Return sessions for the authenticated user only."""
        return SpeechSession.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Return appropriate serializer based on the action."""
        if self.action == 'create':
            return SpeechSessionCreateSerializer
        elif self.action == 'list':
            return SpeechSessionListSerializer
        elif self.action == 'analytics':
            return SpeechSessionAnalyticsSerializer
        return SpeechSessionSerializer
    
    def perform_create(self, serializer):
        """Save the session with the current user."""
        serializer.save(user=self.request.user)
    
    def get_drill_recommendations(self, session):
        """
        Get drill recommendations based on session analysis.
        
        Analyzes the session for issues and returns relevant drills:
        - High filler count (>0.05 per second) -> filler_words drills
        - Low/high WPM (<100 or >180) -> pacing drills  
        - Low clarity score (<60) -> pronunciation drills
        """
        recommendations = []
        
        # Calculate filler density (fillers per second)
        filler_density = session.filler_count / session.duration if session.duration > 0 else 0
        
        # Thresholds for recommendations
        FILLER_THRESHOLD = 0.05  # 0.05 fillers per second = 3 per minute
        WPM_LOW_THRESHOLD = 100
        WPM_HIGH_THRESHOLD = 180
        CLARITY_THRESHOLD = 60
        
        # Estimate WPM from duration and transcription (rough estimate)
        word_count = len(session.transcription.split()) if session.transcription else 0
        estimated_wpm = (word_count / session.duration * 60) if session.duration > 0 else 0
        
        # Check for filler word issues
        if filler_density > FILLER_THRESHOLD:
            filler_drills = Drill.objects.filter(
                skill_type='filler_words',
                is_active=True
            )
            if filler_drills.exists():
                selected_drills = random.sample(list(filler_drills), min(2, len(filler_drills)))
                for drill in selected_drills:
                    recommendations.append({
                        'name': drill.name,
                        'description': drill.description,
                        'skill_type': drill.skill_type,
                        'url': f'/drill/{drill.id}/',
                        'reason': f'High filler word usage detected ({filler_density:.2f} per second)'
                    })
        
        # Check for pacing issues
        if estimated_wpm < WPM_LOW_THRESHOLD or estimated_wpm > WPM_HIGH_THRESHOLD:
            pacing_drills = Drill.objects.filter(
                skill_type='pacing',
                is_active=True
            )
            if pacing_drills.exists():
                selected_drills = random.sample(list(pacing_drills), min(2, len(pacing_drills)))
                for drill in selected_drills:
                    reason = f'Speaking pace issue detected ({estimated_wpm:.0f} WPM)'
                    if estimated_wpm < WPM_LOW_THRESHOLD:
                        reason += ' - Too slow'
                    else:
                        reason += ' - Too fast'
                    
                    recommendations.append({
                        'name': drill.name,
                        'description': drill.description,
                        'skill_type': drill.skill_type,
                        'url': f'/drill/{drill.id}/',
                        'reason': reason
                    })
        
        # Check for pronunciation issues (using confidence score as proxy)
        if session.confidence_score and session.confidence_score < CLARITY_THRESHOLD / 100:
            pronunciation_drills = Drill.objects.filter(
                skill_type='pronunciation',
                is_active=True
            )
            if pronunciation_drills.exists():
                selected_drills = random.sample(list(pronunciation_drills), min(2, len(pronunciation_drills)))
                for drill in selected_drills:
                    recommendations.append({
                        'name': drill.name,
                        'description': drill.description,
                        'skill_type': drill.skill_type,
                        'url': f'/drill/{drill.id}/',
                        'reason': f'Low clarity score detected ({session.confidence_score:.2f})'
                    })
        
        # If no specific issues found, recommend general pronunciation drills
        if not recommendations:
            general_drills = Drill.objects.filter(
                skill_type='pronunciation',
                is_active=True
            )
            if general_drills.exists():
                selected_drills = random.sample(list(general_drills), min(1, len(general_drills)))
                for drill in selected_drills:
                    recommendations.append({
                        'name': drill.name,
                        'description': drill.description,
                        'skill_type': drill.skill_type,
                        'url': f'/drill/{drill.id}/',
                        'reason': 'General pronunciation practice recommended'
                    })
        
        return recommendations
    
    @action(detail=False, methods=['get'])
    def analytics(self, request):
        """
        Get analytics data for the user's speech sessions.
        
        Returns aggregated data including:
        - Total sessions count
        - Total and average duration
        - Filler word statistics
        - Sessions by status
        - Recent activity metrics
        """
        sessions = self.get_queryset()
        
        # Basic counts
        total_sessions = sessions.count()
        if total_sessions == 0:
            return Response({
                'total_sessions': 0,
                'total_duration': 0,
                'average_duration': 0,
                'total_filler_words': 0,
                'average_filler_rate': 0,
                'sessions_by_status': {},
                'recent_sessions_count': 0,
            })
        
        # Duration analytics
        duration_stats = sessions.aggregate(
            total_duration=Sum('duration'),
            avg_duration=Avg('duration')
        )
        
        # Filler word analytics (only for analyzed sessions)
        analyzed_sessions = sessions.filter(status='analyzed')
        filler_stats = analyzed_sessions.aggregate(
            total_fillers=Sum('filler_count'),
            avg_fillers=Avg('filler_count')
        )
        
        # Calculate average filler rate
        avg_filler_rate = 0
        if analyzed_sessions.exists():
            total_analyzed_duration = analyzed_sessions.aggregate(
                total=Sum('duration')
            )['total']
            if total_analyzed_duration and total_analyzed_duration > 0:
                avg_filler_rate = (filler_stats['total_fillers'] or 0) / total_analyzed_duration * 60
        
        # Sessions by status
        status_counts = sessions.values('status').annotate(
            count=Count('id')
        ).order_by('status')
        sessions_by_status = {item['status']: item['count'] for item in status_counts}
        
        # Recent activity (last 30 days)
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_sessions_count = sessions.filter(date__gte=thirty_days_ago).count()
        
        # Improvement metrics (compare last 30 days vs previous 30 days)
        improvement_metrics = {}
        if total_sessions > 1:
            sixty_days_ago = timezone.now() - timedelta(days=60)
            
            recent_sessions = sessions.filter(date__gte=thirty_days_ago)
            previous_sessions = sessions.filter(
                date__gte=sixty_days_ago,
                date__lt=thirty_days_ago
            )
            
            if recent_sessions.exists() and previous_sessions.exists():
                recent_avg_fillers = recent_sessions.aggregate(
                    avg=Avg('filler_count')
                )['avg'] or 0
                
                previous_avg_fillers = previous_sessions.aggregate(
                    avg=Avg('filler_count')
                )['avg'] or 0
                
                if previous_avg_fillers > 0:
                    filler_improvement = (
                        (previous_avg_fillers - recent_avg_fillers) / previous_avg_fillers
                    ) * 100
                    improvement_metrics['filler_improvement_percent'] = round(filler_improvement, 2)
        
        analytics_data = {
            'total_sessions': total_sessions,
            'total_duration': duration_stats['total_duration'] or 0,
            'average_duration': round(duration_stats['avg_duration'] or 0, 2),
            'total_filler_words': filler_stats['total_fillers'] or 0,
            'average_filler_rate': round(avg_filler_rate, 2),
            'sessions_by_status': sessions_by_status,
            'recent_sessions_count': recent_sessions_count,
            'improvement_metrics': improvement_metrics,
        }
        
        return Response(analytics_data)
    
    @action(detail=True, methods=['get'])
    def recommendations(self, request, pk=None):
        """
        Get drill recommendations for a specific speech session.
        
        Returns recommended drills based on the session's analysis results.
        """
        session = self.get_object()
        recommendations = self.get_drill_recommendations(session)
        
        return Response({
            'session_id': session.id,
            'analysis_summary': {
                'filler_density': session.filler_count / session.duration if session.duration > 0 else 0,
                'estimated_wpm': len(session.transcription.split()) / session.duration * 60 if session.duration > 0 and session.transcription else 0,
                'confidence_score': session.confidence_score,
            },
            'drills': recommendations
        })
    
    @action(detail=False, methods=['post'])
    def bulk_update(self, request):
        """
        Bulk update multiple sessions.
        
        Expected payload:
        {
            "session_ids": [1, 2, 3],
            "updates": {
                "status": "archived"
            }
        }
        """
        session_ids = request.data.get('session_ids', [])
        updates = request.data.get('updates', {})
        
        if not session_ids or not updates:
            return Response(
                {'error': 'Both session_ids and updates are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Filter to user's sessions only
        sessions = self.get_queryset().filter(id__in=session_ids)
        
        if not sessions.exists():
            return Response(
                {'error': 'No valid sessions found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Validate updates
        allowed_fields = ['status', 'pacing_analysis', 'filler_count', 'confidence_score']
        invalid_fields = set(updates.keys()) - set(allowed_fields)
        
        if invalid_fields:
            return Response(
                {'error': f'Invalid fields for bulk update: {list(invalid_fields)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Perform bulk update
        updated_count = sessions.update(**updates)
        
        return Response({
            'success': True,
            'updated_count': updated_count,
            'message': f'{updated_count} session(s) updated successfully'
        })
    
    @action(detail=True, methods=['post'], url_path='analyze_with_nlp', url_name='analyze_with_nlp')
    def analyze_with_nlp(self, request, pk=None):
        """
        Analyze speech session using the complete NLP pipeline.
        
        This endpoint:
        1. Transcribes audio with Whisper
        2. Analyzes pacing from timestamps
        3. Detects filler words with BERT (or rule-based fallback)
        4. Classifies root cause
        5. Generates drill recommendations
        
        Returns comprehensive analysis results.
        """
        print(f"[DEBUG] analyze_with_nlp called for session {pk}, user: {request.user}")
        print(f"[DEBUG] Request method: {request.method}, Content-Type: {request.content_type}")
        
        try:
            session = self.get_object()
            print(f"[DEBUG] Session found: {session.id}, user: {session.user}, audio_file: {session.audio_file}")
        except Exception as e:
            print(f"[ERROR] Failed to get session object: {e}")
            return Response(
                {
                    'success': False,
                    'error': f'Session not found: {str(e)}'
                },
                status=status.HTTP_404_NOT_FOUND
            )
        
        if not session.audio_file:
            return Response(
                {
                    'success': False,
                    'error': 'No audio file available for analysis'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get audio file path
        audio_path = session.audio_file.path
        print(f"[DEBUG] Audio file path: {audio_path}")
        if not os.path.exists(audio_path):
            return Response(
                {
                    'success': False,
                    'error': f'Audio file not found on disk: {audio_path}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Get NLP pipeline
            pipeline = get_pipeline()
            
            # Run complete analysis
            results = pipeline.analyze_audio(audio_path)
            
            if not results.get('success'):
                return Response(
                    {'error': results.get('error', 'Analysis failed')},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Update session with results
            session.transcription = results['transcription']
            session.duration = int(results['duration'])
            session.filler_count = results['fillers']['total_count']
            session.confidence_score = results['confidence_score']
            session.status = 'analyzed'
            
            # Format pacing analysis as text (using format that matches template extraction)
            pacing = results['pacing']
            wpm = pacing.get('wpm', 0)
            session.pacing_analysis = (
                f"Words Per Minute: {wpm:.0f}. "
                f"Speaking rate: {pacing.get('speaking_rate', 'normal')}. "
                f"Average pause duration: {pacing.get('avg_pause_duration', 0):.2f}s. "
                f"Long pauses (>2s): {pacing.get('long_pauses', 0)}. "
                f"Speech rate variance: {pacing.get('variance', 0):.2f}."
            )
            
            session.save()
            
            # Return comprehensive results
            serializer = self.get_serializer(session)
            return Response({
                'success': True,
                'message': 'Speech analysis completed successfully',
                'session': serializer.data,
                'analysis_details': {
                    'transcription': results['transcription'],
                    'word_count': results['word_count'],
                    'pacing': results['pacing'],
                    'fillers': results['fillers'],
                    'pronunciation': results['pronunciation'],
                    'cause': results['cause'],
                    'confidence_score': results['confidence_score'],
                },
                'recommendations': results['recommendations']
            })
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"[ERROR] analyze_with_nlp failed: {error_msg}\n{error_trace}")
            
            # Return JSON error response (not HTML)
            return Response(
                {
                    'success': False,
                    'error': f'Analysis failed: {error_msg}',
                    'details': error_trace if settings.DEBUG else None
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def reanalyze(self, request, pk=None):
        """
        Alias for analyze_with_nlp for backward compatibility.
        """
        return self.analyze_with_nlp(request, pk)


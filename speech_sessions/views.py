from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, Http404
from django.core.paginator import Paginator
from django.db.models import Q
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django_otp.decorators import otp_required
from django.utils import timezone
from datetime import datetime
import json
import logging

from .models import SpeechSession
from .forms import SpeechSessionCreateForm, SpeechSessionUpdateForm, SpeechSessionFilterForm

# Setup logging
logger = logging.getLogger(__name__)


def require_2fa_if_enabled(view_func):
    """
    Custom decorator to require 2FA verification if user has 2FA enabled.
    This integrates with the existing 2FA system from the coach app.
    """
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated and request.user.is_2fa_enabled:
            if not request.user.has_2fa_setup():
                messages.warning(request, 'Please complete 2FA setup to access speech sessions.')
                return redirect('coach:setup_2fa')
            # For now, if user is logged in and has 2FA enabled, allow access
            # The 2FA verification should have been completed during login
            pass
        return view_func(request, *args, **kwargs)
    return wrapper


@login_required
@require_2fa_if_enabled
def session_list_view(request):
    """Display list of user's speech sessions with filtering options."""
    
    # Get filter form
    filter_form = SpeechSessionFilterForm(request.GET)
    
    # Start with user's sessions
    sessions = SpeechSession.objects.filter(user=request.user)
    
    # Apply filters if form is valid
    if filter_form.is_valid():
        status = filter_form.cleaned_data.get('status')
        date_from = filter_form.cleaned_data.get('date_from')
        date_to = filter_form.cleaned_data.get('date_to')
        search = filter_form.cleaned_data.get('search')
        
        if status:
            sessions = sessions.filter(status=status)
        
        if date_from:
            sessions = sessions.filter(date__date__gte=date_from)
        
        if date_to:
            sessions = sessions.filter(date__date__lte=date_to)
        
        if search:
            sessions = sessions.filter(
                Q(transcription__icontains=search) | 
                Q(pacing_analysis__icontains=search)
            )
    
    # Order by date (most recent first)
    sessions = sessions.order_by('-date')
    
    # Pagination
    paginator = Paginator(sessions, 10)  # Show 10 sessions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Calculate statistics
    total_sessions = SpeechSession.objects.filter(user=request.user).count()
    pending_sessions = SpeechSession.objects.filter(user=request.user, status='pending').count()
    analyzed_sessions = SpeechSession.objects.filter(user=request.user, status='analyzed').count()
    
    context = {
        'page_obj': page_obj,
        'filter_form': filter_form,
        'total_sessions': total_sessions,
        'pending_sessions': pending_sessions,
        'analyzed_sessions': analyzed_sessions,
    }
    
    return render(request, 'speech_sessions/session_list.html', context)


@login_required
@require_2fa_if_enabled
def session_create_view(request):
    """Create a new speech session."""
    
    if request.method == 'POST':
        form = SpeechSessionCreateForm(request.POST, request.FILES)
        if form.is_valid():
            session = form.save(commit=False)
            session.user = request.user
            
            # Handle recorded audio from browser (base64)
            recorded_audio = request.POST.get('recorded_audio', '').strip()
            if recorded_audio and not session.audio_file:
                import base64
                import tempfile
                import os
                from django.core.files.base import ContentFile
                from django.core.files.uploadedfile import InMemoryUploadedFile
                import io
                
                try:
                    # Remove data URL prefix if present (e.g., "data:audio/webm;base64,")
                    if ',' in recorded_audio:
                        recorded_audio = recorded_audio.split(',')[1]
                    
                    # Decode base64 audio
                    audio_data = base64.b64decode(recorded_audio)
                    
                    # Create a file-like object from the decoded audio
                    # Use webm format since that's what MediaRecorder produces
                    audio_file = ContentFile(audio_data, name='recording.webm')
                    
                    # Save to session
                    session.audio_file.save('recording.webm', audio_file, save=False)
                    
                    logger.info("Recorded audio converted and saved to session")
                except Exception as e:
                    logger.error(f"Failed to process recorded audio: {e}")
                    messages.error(request, 'Failed to process recorded audio. Please try again.')
                    return render(request, 'speech_sessions/session_form.html', {
                        'form': form,
                        'title': 'Create New Speech Session',
                        'submit_text': 'Create Session'
                    })
            
            # Calculate duration from uploaded audio file
            # Note: form.save(commit=False) doesn't save the file to disk yet
            # We need to save the file temporarily to calculate duration
            if session.audio_file:
                import librosa
                import os
                import tempfile
                
                temp_file_path = None
                
                try:
                    # Get the uploaded file from request.FILES
                    audio_file = request.FILES.get('audio_file')
                    if not audio_file:
                        # Fallback: try to get from session.audio_file
                        audio_file = session.audio_file
                    
                    # Save to temporary location to calculate duration
                    # Get file extension from original filename
                    file_extension = os.path.splitext(audio_file.name)[1] if hasattr(audio_file, 'name') else '.mp3'
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        # Write uploaded file to temporary location
                        for chunk in audio_file.chunks():
                            temp_file.write(chunk)
                        temp_file_path = temp_file.name
                    
                    # Calculate duration using librosa (primary method)
                    # librosa.load with sr=None doesn't resample, just gets duration
                    y, sr = librosa.load(temp_file_path, sr=None)
                    duration_seconds = len(y) / sr
                    session.duration = int(round(duration_seconds))
                    
                    logger.info(f"Duration calculated: {session.duration} seconds for {audio_file.name if hasattr(audio_file, 'name') else 'audio file'}")
                    
                except Exception as e:
                    # Fallback: try to get duration from file metadata using mutagen
                    logger.warning(f"Could not calculate duration with librosa: {e}")
                    
                    # Try mutagen as fallback (only if we have temp_file_path)
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            import mutagen
                            from mutagen import File as MutagenFile
                            
                            audio = MutagenFile(temp_file_path)
                            if audio is not None and hasattr(audio, 'info'):
                                duration_seconds = audio.info.length
                                session.duration = int(round(duration_seconds))
                                logger.info(f"Duration calculated using mutagen: {session.duration} seconds")
                            else:
                                session.duration = 0
                                logger.error(f"Could not determine duration from metadata for {temp_file_path}")
                        except ImportError:
                            logger.warning("mutagen not available, cannot use fallback method")
                            session.duration = 0
                        except Exception as e2:
                            logger.error(f"Mutagen fallback also failed: {e2}")
                            session.duration = 0
                    else:
                        logger.error("Could not access audio file for duration calculation")
                        session.duration = 0
                
                finally:
                    # Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                
                # Save session first to get audio file path
                session.save()
                
                # Run real analysis pipeline
                try:
                    from coach.utils import transcribe_audio, analyze_speech_pipeline
                    from speech_sessions.nlp_pipeline import get_pipeline
                    
                    # Get audio file path
                    audio_path = session.audio_file.path if session.audio_file else None
                    
                    if audio_path and os.path.exists(audio_path):
                        # Transcribe audio
                        logger.info(f"Transcribing audio: {audio_path}")
                        transcription_result = transcribe_audio(audio_path)
                        
                        if transcription_result.get('success') and transcription_result.get('text'):
                            text = transcription_result['text'].strip()
                            session.transcription = text
                            
                            # Run full analysis pipeline
                            logger.info(f"Running analysis pipeline for session {session.pk}")
                            pipeline_result = analyze_speech_pipeline(
                                audio_path=audio_path,
                                transcript=text,
                                duration_seconds=session.duration
                            )
                            
                            if pipeline_result.get('success'):
                                # Extract results
                                quality = pipeline_result.get('quality', {})
                                bert_scores = pipeline_result.get('bert_scores', {})
                                
                                # Use NLP pipeline for enhanced filler detection with tagging
                                logger.info(f"Detecting and tagging filler words for session {session.pk}")
                                nlp_pipeline = get_pipeline()
                                
                                # Get word segments from transcription if available
                                segments = transcription_result.get('segments', [])
                                
                                # Detect fillers with NLP (spaCy/regex) and tagging
                                filler_analysis = nlp_pipeline.detect_fillers(text, segments)
                                
                                # Save detected fillers to database
                                saved_fillers_count = nlp_pipeline.save_detected_fillers(session, filler_analysis)
                                logger.info(f"Saved {saved_fillers_count} tagged filler words to database")
                                
                                # Calculate filler count from NLP analysis
                                session.filler_count = filler_analysis.get('total_count', 0)
                                
                                # Store pacing analysis
                                wpm = quality.get('wpm', 0)
                                pacing_score_raw = quality.get('pacing_score', 0)
                                session.pacing_analysis = f"Speaking rate: {wpm:.1f} WPM. Pacing score: {pacing_score_raw:.1f}/100."
                                
                                # Calculate confidence score
                                session.confidence_score = quality.get('confidence_score', 0.0)
                                
                                # Store ML model scores (0.0-1.0, higher = worse performance)
                                session.filler_score = bert_scores.get('filler')
                                session.clarity_score = bert_scores.get('clarity')
                                session.pacing_score = bert_scores.get('pacing')
                                
                                session.status = 'analyzed'
                                logger.info(f"Analysis complete for session {session.pk}: {session.filler_count} fillers ({saved_fillers_count} saved), {wpm:.1f} WPM, scores - F:{session.filler_score:.3f}, C:{session.clarity_score:.3f}, P:{session.pacing_score:.3f}")
                            else:
                                # Analysis failed but transcription succeeded
                                logger.warning(f"Analysis pipeline failed for session {session.pk}: {pipeline_result.get('error')}")
                                session.status = 'pending'
                        else:
                            # Transcription failed
                            error_msg = transcription_result.get('error', 'Transcription failed')
                            logger.error(f"Transcription failed for session {session.pk}: {error_msg}")
                            session.status = 'pending'
                    else:
                        logger.warning(f"Audio file not found for session {session.pk}")
                        session.status = 'pending'
                        
                except Exception as e:
                    logger.error(f"Error during analysis for session {session.pk}: {e}", exc_info=True)
                    session.status = 'pending'  # Keep as pending if analysis fails
                
                # Save session with analysis results
                session.save()
                
            else:
                # No audio file uploaded - set duration to 0 as fallback
                session.duration = 0
                session.status = 'pending'  # No analysis without audio file
            session.save()
            
            messages.success(request, 'Speech session created successfully!')
            return redirect('speech_sessions:session_detail', pk=session.pk)
    else:
        form = SpeechSessionCreateForm()
    
    return render(request, 'speech_sessions/session_form.html', {
        'form': form,
        'title': 'Create New Speech Session',
        'submit_text': 'Create Session'
    })


@login_required
@require_2fa_if_enabled
def session_detail_view(request, pk):
    """Display detailed view of a speech session with drill recommendations."""
    
    session = get_object_or_404(SpeechSession, pk=pk, user=request.user)
    
    # Get drill recommendations if session is analyzed
    recommended_drill = None
    ensemble_scores = None
    detailed_feedback = None
    drill_reason = None
    
    # Metrics used for explanations
    filler_density_actual = None
    fillers_per_min = None
    good_clarity_percentage = None
    wpm_calculated = None
    
    # Generate detailed feedback and drill recommendations if session is analyzed
    # Check if we have minimum required data (need transcription for ensemble scores)
    if session.status == 'analyzed' and session.transcription:
        try:
            from coach.utils import get_bert_scores, predict_cause, predict_cause_fallback, DrillRecommender
            import os
            # Get ensemble scores (requires transcription)
            audio_path = session.audio_file.path if session.audio_file and os.path.exists(session.audio_file.path) else None
            ensemble_scores = get_bert_scores(session.transcription, audio_path=audio_path)
            
            # Get cause classification
            cause_result = predict_cause(audio_path=audio_path, transcript=session.transcription, duration=session.duration)
            
            # Calculate actual metrics for cross-validation
            word_count = len(session.transcription.split())
            duration_minutes = session.duration / 60.0 if session.duration > 0 else 0
            wpm_calculated = int((word_count / duration_minutes) if duration_minutes > 0 else 0)
            filler_count_actual = session.filler_count if hasattr(session, 'filler_count') else 0
            filler_density_actual = (filler_count_actual / word_count * 100) if word_count > 0 else 0
            fillers_per_min = (filler_count_actual / duration_minutes) if duration_minutes > 0 else 0
            
            # Pass actual metrics to recommender for cross-validation
            ensemble_scores_with_metrics = ensemble_scores.copy()
            ensemble_scores_with_metrics['_actual_metrics'] = {
                'wpm': wpm_calculated,
                'filler_count': filler_count_actual,
                'filler_density': filler_density_actual
            }
            
            # Get drill recommendations with actual metrics
            recommender = DrillRecommender()
            history = list(SpeechSession.objects.filter(user=request.user).order_by('-date')[:7])
            recommended_drill = recommender.pick_best(ensemble_scores_with_metrics, history)

            # Generate detailed feedback using ACTUAL METRICS, not raw BERT scores
            # Use actual metrics to show true performance
            filler_score_display = filler_density_actual / 100.0 if filler_density_actual > 0 else ensemble_scores.get('filler', 0.0)
            clarity_score = ensemble_scores.get('clarity', 0.0)
            pacing_score = ensemble_scores.get('pacing', 0.0)
            
            # Adjust pacing score based on actual WPM
            if wpm_calculated > 0:
                if wpm_calculated < 120:
                    pacing_score_display = min(1.0, pacing_score + 0.3)  # Too slow = worse
                elif wpm_calculated > 180:
                    pacing_score_display = min(1.0, pacing_score + 0.2)  # Too fast = worse
                else:
                    pacing_score_display = max(0.0, pacing_score - 0.2)  # Optimal = better
            else:
                pacing_score_display = pacing_score
            
            # Use adjusted scores for display and downstream logic
            filler_score = filler_score_display
            pacing_score = pacing_score_display
            ensemble_scores['filler'] = filler_score
            ensemble_scores['pacing'] = pacing_score
            
            # Calculate good clarity percentage for consistency with top section
            # BERT clarity_score is "bad clarity" (higher = worse), convert to "good clarity" (higher = better)
            good_clarity_percentage = max(0, (1.0 - clarity_score) * 100)
            
            # Store good clarity score in ensemble_scores for template use
            ensemble_scores['good_clarity_percentage'] = good_clarity_percentage
            
            feedback_parts = []
            
            # Filler feedback - use ACTUAL filler metrics, not BERT score
            filler_percentage = filler_density_actual if filler_density_actual > 0 else (filler_score * 100)
            
            if filler_percentage > 6.0 or fillers_per_min > 10:
                feedback_parts.append(f"⚠️ HIGH FILLER USAGE ({filler_percentage:.1f}%): You used many filler words ({filler_count_actual} total, {fillers_per_min:.1f}/min). Focus on pausing instead of saying 'um' or 'uh'.")
            elif filler_percentage > 3.0 or fillers_per_min > 5:
                feedback_parts.append(f"⚠️ MODERATE FILLER USAGE ({filler_percentage:.1f}%): Some filler words detected ({filler_count_actual} total, {fillers_per_min:.1f}/min). Practice awareness of your speech patterns.")
            else:
                feedback_parts.append(f"✅ EXCELLENT FILLER CONTROL ({filler_percentage:.1f}%): Minimal filler words detected! ({filler_count_actual} total, {fillers_per_min:.1f}/min)")
            
            # Clarity feedback - convert to "good clarity" percentage (higher = better)
            # BERT clarity_score is "bad clarity" (higher = worse), so convert: good_clarity = 100 - (bad_clarity * 100)
            good_clarity_percentage = max(0, (1.0 - clarity_score) * 100)
            
            if good_clarity_percentage >= 80:
                feedback_parts.append(f"✅ EXCELLENT CLARITY ({good_clarity_percentage:.0f}%): Clear and articulate speech! Your pronunciation is clear and articulate.")
            elif good_clarity_percentage >= 60:
                feedback_parts.append(f"⚠️ GOOD CLARITY ({good_clarity_percentage:.0f}%): Good clarity with room for improvement. Focus on enunciation.")
            else:
                feedback_parts.append(f"⚠️ CLARITY NEEDS IMPROVEMENT ({good_clarity_percentage:.0f}%): Speech clarity could be enhanced. Focus on clear pronunciation and articulation.")
            
            # Pacing feedback - use ACTUAL WPM, not just BERT score
            if wpm_calculated > 0:
                if wpm_calculated < 120:
                    feedback_parts.append(f"⚠️ PACING NEEDS WORK ({wpm_calculated} WPM): Your speaking rate is too slow. Aim for 120-180 WPM to maintain engagement.")
                elif wpm_calculated > 180:
                    feedback_parts.append(f"⚠️ PACING NEEDS WORK ({wpm_calculated} WPM): Your speaking rate is too fast. Slow down to 120-180 WPM for better clarity.")
                else:
                    feedback_parts.append(f"✅ EXCELLENT PACING ({wpm_calculated} WPM): Consistent and well-paced speech! You're in the optimal 120-180 WPM range.")
            else:
                # Fallback to BERT score if WPM not available
                if pacing_score > 0.6:
                    feedback_parts.append(f"⚠️ PACING NEEDS WORK ({pacing_score*100:.0f}%): Your speaking rate may be too fast or irregular. Focus on consistent pacing.")
                elif pacing_score > 0.4:
                    feedback_parts.append(f"⚠️ MODERATE PACING ({pacing_score*100:.0f}%): Good pacing with occasional variations. Practice maintaining steady rhythm.")
                else:
                    feedback_parts.append(f"✅ EXCELLENT PACING ({pacing_score*100:.0f}%): Consistent and well-paced speech!")
            
            # Cause-specific feedback
            cause = cause_result.get('cause', 'other')
            cause_feedback = {
                'lexical': "Focus on word choice and vocabulary expansion.",
                'syntactic': "Work on sentence structure and grammar clarity.",
                'articulatory': "Practice pronunciation and articulation exercises.",
                'fluency': "Focus on reducing hesitations and improving flow.",
                'other': "General speech improvement recommended."
            }
            feedback_parts.append(f"📊 ROOT CAUSE: {cause_feedback.get(cause, 'Other')}")
            
            detailed_feedback = "\n\n".join(feedback_parts)
            
            # Build drill explanation aligned with recommended drill
            if recommended_drill:
                if recommended_drill.skill_type == 'filler_words':
                    drill_reason = (
                        "This drill targets your filler control "
                        f"({filler_percentage:.1f}% filler density, {fillers_per_min:.1f} per minute)."
                    )
                elif recommended_drill.skill_type == 'pronunciation':
                    if good_clarity_percentage is not None:
                        drill_reason = (
                            "This drill focuses on speech clarity "
                            f"({good_clarity_percentage:.0f}% clarity score)."
                        )
                    else:
                        drill_reason = "This drill sharpens pronunciation based on your recent analysis."
                elif recommended_drill.skill_type == 'pacing':
                    if wpm_calculated:
                        drill_reason = (
                            "This drill helps stabilize your pacing "
                            f"({wpm_calculated} words per minute)."
                        )
                    else:
                        drill_reason = "This drill reinforces consistent pacing habits."
                else:
                    drill_reason = "This drill strengthens overall speaking performance identified in your analysis."
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to get recommendations: {e}\n{traceback.format_exc()}")
            # Continue without recommendations
    
    # Extract WPM from pacing_analysis if available (handle both formats)
    wpm = None
    if session.pacing_analysis:
        import re
        # Try "Words Per Minute: X" first
        wpm_match = re.search(r'Words Per Minute:\s*(\d+)', session.pacing_analysis)
        if not wpm_match:
            # Fallback to "WPM: X" format
            wpm_match = re.search(r'WPM:\s*(\d+)', session.pacing_analysis)
        if wpm_match:
            wpm = int(wpm_match.group(1))
        # If still no match, try to calculate from duration and transcription
        if not wpm and session.duration > 0 and session.transcription:
            word_count = len(session.transcription.split())
            wpm = int((word_count / session.duration) * 60)
    
    # Calculate consistent clarity score for display in top section
    # If we have ensemble_scores, use the converted good_clarity_percentage
    # Otherwise, fall back to session.confidence_score if available
    display_clarity_score = None
    if ensemble_scores and 'good_clarity_percentage' in ensemble_scores:
        display_clarity_score = ensemble_scores['good_clarity_percentage'] / 100.0  # Convert to 0-1 range for template
    elif session.confidence_score:
        display_clarity_score = session.confidence_score
    
    return render(request, 'speech_sessions/session_detail.html', {
        'session': session,
        'recommended_drill': recommended_drill,
        'ensemble_scores': ensemble_scores,
        'detailed_feedback': detailed_feedback,
        'wpm': wpm,
        'display_clarity_score': display_clarity_score,  # Consistent clarity score for top section
        'drill_reason': drill_reason,
    })


@login_required
@require_2fa_if_enabled
def session_update_view(request, pk):
    """Update an existing speech session."""
    
    session = get_object_or_404(SpeechSession, pk=pk, user=request.user)
    
    if request.method == 'POST':
        form = SpeechSessionUpdateForm(request.POST, instance=session)
        if form.is_valid():
            form.save()
            messages.success(request, 'Speech session updated successfully!')
            return redirect('speech_sessions:session_detail', pk=session.pk)
    else:
        form = SpeechSessionUpdateForm(instance=session)
    
    return render(request, 'speech_sessions/session_form.html', {
        'form': form,
        'session': session,
        'title': 'Update Speech Session',
        'submit_text': 'Update Session'
    })


@login_required
@require_2fa_if_enabled
def session_delete_view(request, pk):
    """Delete a speech session with confirmation."""
    
    session = get_object_or_404(SpeechSession, pk=pk, user=request.user)
    
    if request.method == 'POST':
        session_id = session.id
        session.delete()
        messages.success(request, f'Speech session #{session_id} deleted successfully!')
        return redirect('speech_sessions:session_list')
    
    return render(request, 'speech_sessions/session_confirm_delete.html', {
        'session': session
    })


@login_required
@require_2fa_if_enabled
@require_http_methods(["POST"])
def session_bulk_action_view(request):
    """Handle bulk actions on multiple sessions."""
    
    try:
        data = json.loads(request.body)
        action = data.get('action')
        session_ids = data.get('session_ids', [])
        
        if not action or not session_ids:
            return JsonResponse({'error': 'Missing action or session IDs'}, status=400)
        
        # Ensure user can only act on their own sessions
        sessions = SpeechSession.objects.filter(
            id__in=session_ids, 
            user=request.user
        )
        
        count = sessions.count()
        if count == 0:
            return JsonResponse({'error': 'No valid sessions found'}, status=404)
        
        if action == 'delete':
            sessions.delete()
            return JsonResponse({
                'success': True, 
                'message': f'{count} session(s) deleted successfully'
            })
        elif action == 'archive':
            sessions.update(status='archived')
            return JsonResponse({
                'success': True, 
                'message': f'{count} session(s) archived successfully'
            })
        elif action == 'mark_analyzed':
            sessions.update(status='analyzed')
            return JsonResponse({
                'success': True, 
                'message': f'{count} session(s) marked as analyzed'
            })
        else:
            return JsonResponse({'error': 'Invalid action'}, status=400)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_2fa_if_enabled
def session_analytics_view(request):
    """Display analytics and insights for user's speech sessions."""
    
    sessions = SpeechSession.objects.filter(user=request.user)
    
    # Calculate analytics
    total_sessions = sessions.count()
    total_duration = sum(session.duration for session in sessions)
    avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
    
    # Filler word analysis
    analyzed_sessions = sessions.filter(status='analyzed')
    total_fillers = sum(session.filler_count for session in analyzed_sessions)
    avg_fillers = total_fillers / analyzed_sessions.count() if analyzed_sessions.count() > 0 else 0
    
    # Recent progress (last 30 days)
    thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
    recent_sessions = sessions.filter(date__gte=thirty_days_ago)
    
    context = {
        'total_sessions': total_sessions,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'total_fillers': total_fillers,
        'avg_fillers': avg_fillers,
        'recent_sessions_count': recent_sessions.count(),
        'analyzed_sessions_count': analyzed_sessions.count(),
        'pending_sessions_count': sessions.filter(status='pending').count(),
        'archived_sessions_count': sessions.filter(status='archived').count(),
    }
    
    return render(request, 'speech_sessions/session_analytics.html', context)
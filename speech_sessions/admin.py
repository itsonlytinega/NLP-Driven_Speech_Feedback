from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import SpeechSession, DetectedFiller


@admin.register(SpeechSession)
class SpeechSessionAdmin(admin.ModelAdmin):
    """
    Admin interface for managing speech sessions.
    
    Provides comprehensive management capabilities for administrators
    to view, edit, and analyze speech sessions.
    """
    
    list_display = [
        'id',
        'user_link',
        'date',
        'duration_display',
        'filler_count',
        'status_badge',
        'confidence_display',
        'has_audio',
        'created_at',
    ]
    
    list_filter = [
        'status',
        'date',
        'created_at',
        ('user', admin.RelatedOnlyFieldListFilter),
    ]
    
    search_fields = [
        'user__email',
        'user__first_name',
        'user__last_name',
        'transcription',
        'pacing_analysis',
    ]
    
    readonly_fields = [
        'id',
        'date',
        'created_at',
        'updated_at',
        'duration_minutes',
        'filler_rate',
        'audio_player',
    ]
    
    fieldsets = (
        ('Basic Information', {
            'fields': (
                'id',
                'user',
                'date',
                'status',
            )
        }),
        ('Session Details', {
            'fields': (
                'duration',
                'duration_minutes',
                'audio_file',
                'audio_player',
            )
        }),
        ('Analysis Results', {
            'fields': (
                'filler_count',
                'filler_rate',
                'pacing_analysis',
                'confidence_score',
                'transcription',
            )
        }),
        ('Timestamps', {
            'fields': (
                'created_at',
                'updated_at',
            ),
            'classes': ('collapse',)
        }),
    )
    
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    list_per_page = 25
    
    actions = [
        'mark_as_analyzed',
        'mark_as_pending',
        'mark_as_archived',
        'export_sessions',
    ]
    
    def user_link(self, obj):
        """Display user with link to their profile."""
        url = reverse('admin:coach_user_change', args=[obj.user.pk])
        return format_html('<a href="{}">{}</a>', url, obj.user.email)
    user_link.short_description = 'User'
    user_link.admin_order_field = 'user__email'
    
    def duration_display(self, obj):
        """Display duration in a user-friendly format."""
        return obj.duration_minutes
    duration_display.short_description = 'Duration'
    duration_display.admin_order_field = 'duration'
    
    def status_badge(self, obj):
        """Display status as a colored badge."""
        colors = {
            'pending': '#FCD34D',  # yellow
            'analyzed': '#10B981', # green
            'archived': '#6B7280', # gray
        }
        color = colors.get(obj.status, '#6B7280')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 12px; font-size: 11px; font-weight: bold;">{}</span>',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    status_badge.admin_order_field = 'status'
    
    def confidence_display(self, obj):
        """Display confidence score as percentage."""
        if obj.confidence_score is not None:
            percentage = int(obj.confidence_score * 100)
            color = '#10B981' if percentage >= 80 else '#F59E0B' if percentage >= 60 else '#EF4444'
            return format_html(
                '<span style="color: {}; font-weight: bold;">{}%</span>',
                color,
                percentage
            )
        return '-'
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence_score'
    
    def has_audio(self, obj):
        """Display whether session has audio file."""
        if obj.audio_file:
            return format_html(
                '<span style="color: #10B981;"><i class="fas fa-check-circle"></i></span>'
            )
        return format_html(
            '<span style="color: #EF4444;"><i class="fas fa-times-circle"></i></span>'
        )
    has_audio.short_description = 'Audio'
    has_audio.admin_order_field = 'audio_file'
    
    def audio_player(self, obj):
        """Display audio player if audio file exists."""
        if obj.audio_file:
            return format_html(
                '<audio controls style="width: 300px;">'
                '<source src="{}" type="audio/mpeg">'
                '<source src="{}" type="audio/wav">'
                'Your browser does not support the audio element.'
                '</audio>',
                obj.audio_file.url,
                obj.audio_file.url
            )
        return 'No audio file'
    audio_player.short_description = 'Audio Player'
    
    def mark_as_analyzed(self, request, queryset):
        """Mark selected sessions as analyzed."""
        updated = queryset.update(status='analyzed')
        self.message_user(
            request,
            f'{updated} session(s) marked as analyzed.'
        )
    mark_as_analyzed.short_description = 'Mark selected sessions as analyzed'
    
    def mark_as_pending(self, request, queryset):
        """Mark selected sessions as pending."""
        updated = queryset.update(status='pending')
        self.message_user(
            request,
            f'{updated} session(s) marked as pending.'
        )
    mark_as_pending.short_description = 'Mark selected sessions as pending'
    
    def mark_as_archived(self, request, queryset):
        """Mark selected sessions as archived."""
        updated = queryset.update(status='archived')
        self.message_user(
            request,
            f'{updated} session(s) marked as archived.'
        )
    mark_as_archived.short_description = 'Mark selected sessions as archived'
    
    def export_sessions(self, request, queryset):
        """Export selected sessions (placeholder)."""
        count = queryset.count()
        self.message_user(
            request,
            f'Export functionality for {count} session(s) would be implemented here.'
        )
    export_sessions.short_description = 'Export selected sessions'
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('user')
    
    class Media:
        css = {
            'all': ('admin/css/speech_sessions.css',)  # Custom CSS if needed
        }
        js = ('admin/js/speech_sessions.js',)  # Custom JS if needed


@admin.register(DetectedFiller)
class DetectedFillerAdmin(admin.ModelAdmin):
    """
    Admin interface for managing detected filler words.
    
    Provides view and management capabilities for individual filler word detections.
    """
    
    list_display = [
        'id',
        'speech_session_link',
        'filler_word',
        'word_position',
        'detection_method_badge',
        'confidence_display',
        'start_time',
        'end_time',
        'created_at',
    ]
    
    list_filter = [
        'detection_method',
        'filler_word',
        'created_at',
        ('speech_session', admin.RelatedOnlyFieldListFilter),
    ]
    
    search_fields = [
        'filler_word',
        'speech_session__user__email',
        'context_before',
        'context_after',
    ]
    
    readonly_fields = [
        'id',
        'created_at',
    ]
    
    fieldsets = (
        ('Basic Information', {
            'fields': (
                'id',
                'speech_session',
                'filler_word',
                'word_position',
            )
        }),
        ('Timing Information', {
            'fields': (
                'start_time',
                'end_time',
            )
        }),
        ('Context', {
            'fields': (
                'context_before',
                'context_after',
            )
        }),
        ('Detection Details', {
            'fields': (
                'detection_method',
                'confidence',
            )
        }),
        ('Timestamps', {
            'fields': (
                'created_at',
            ),
            'classes': ('collapse',)
        }),
    )
    
    ordering = ['-created_at', 'speech_session', 'word_position']
    date_hierarchy = 'created_at'
    list_per_page = 50
    
    def speech_session_link(self, obj):
        """Display speech session with link."""
        url = reverse('admin:speech_sessions_speechsession_change', args=[obj.speech_session.pk])
        return format_html('<a href="{}">Session #{}</a>', url, obj.speech_session.pk)
    speech_session_link.short_description = 'Speech Session'
    speech_session_link.admin_order_field = 'speech_session'
    
    def detection_method_badge(self, obj):
        """Display detection method as a colored badge."""
        colors = {
            'spacy': '#3B82F6',  # blue
            'regex': '#10B981',  # green
            'bert': '#8B5CF6',   # purple
            'rule_based': '#6B7280',  # gray
        }
        color = colors.get(obj.detection_method, '#6B7280')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 12px; font-size: 11px; font-weight: bold;">{}</span>',
            color,
            obj.get_detection_method_display()
        )
    detection_method_badge.short_description = 'Method'
    detection_method_badge.admin_order_field = 'detection_method'
    
    def confidence_display(self, obj):
        """Display confidence score as percentage."""
        percentage = int(obj.confidence * 100)
        color = '#10B981' if percentage >= 80 else '#F59E0B' if percentage >= 60 else '#EF4444'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}%</span>',
            color,
            percentage
        )
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence'
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('speech_session', 'speech_session__user')
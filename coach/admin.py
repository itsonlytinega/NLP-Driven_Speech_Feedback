from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Feedback, Drill, DrillCompletion, EmailOTP, WebAuthnCredential


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'is_email_verified', 'is_2fa_enabled', 'is_staff', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'is_email_verified', 'is_2fa_enabled', 'date_joined')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone_number')}),
        ('Security', {'fields': ('is_email_verified', 'is_2fa_enabled', 'backup_codes')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'f1_score', 'fillers_count', 'wpm', 'created_at')
    list_filter = ('created_at', 'f1_score')
    search_fields = ('user__email',)
    ordering = ('-created_at',)


@admin.register(Drill)
class DrillAdmin(admin.ModelAdmin):
    list_display = ('name', 'skill_type', 'cause', 'is_active', 'created_at')
    list_filter = ('skill_type', 'cause', 'is_active', 'created_at')
    search_fields = ('name', 'description', 'skill_type')
    ordering = ('skill_type', 'name')
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'skill_type', 'cause', 'description', 'is_active')
        }),
        ('Interactive Elements', {
            'fields': ('interactive_elements',),
            'classes': ('collapse',)
        }),
    )


@admin.register(DrillCompletion)
class DrillCompletionAdmin(admin.ModelAdmin):
    list_display = ('user', 'drill', 'score', 'duration_seconds', 'completed_at')
    list_filter = ('drill__skill_type', 'completed_at', 'score')
    search_fields = ('user__email', 'drill__name', 'notes')
    ordering = ('-completed_at',)
    readonly_fields = ('completed_at',)


@admin.register(EmailOTP)
class EmailOTPAdmin(admin.ModelAdmin):
    list_display = ('user', 'code', 'is_used', 'is_expired', 'created_at', 'expires_at')
    list_filter = ('is_used', 'created_at', 'expires_at')
    search_fields = ('user__email', 'code')
    ordering = ('-created_at',)
    readonly_fields = ('code', 'created_at', 'expires_at')
    
    def is_expired(self, obj):
        return obj.is_expired()
    is_expired.boolean = True
    is_expired.short_description = 'Expired'


@admin.register(WebAuthnCredential)
class WebAuthnCredentialAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'device_type', 'is_active', 'last_used', 'created_at')
    list_filter = ('is_active', 'device_type', 'created_at', 'last_used')
    search_fields = ('user__email', 'name', 'credential_id')
    ordering = ('-created_at',)
    readonly_fields = ('credential_id', 'public_key', 'sign_count', 'created_at', 'updated_at', 'last_used')
    
    fieldsets = (
        ('Credential Info', {
            'fields': ('user', 'name', 'credential_id', 'is_active')
        }),
        ('Technical Details', {
            'fields': ('public_key', 'sign_count', 'credential_type', 'aaguid', 'transports'),
            'classes': ('collapse',)
        }),
        ('Device Info', {
            'fields': ('device_type', 'backup_eligible', 'backup_state')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_used')
        }),
    )


from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Feedback, Drill, EmailOTP


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
    list_display = ('type', 'instruction', 'is_active', 'created_at')
    list_filter = ('type', 'is_active', 'created_at')
    search_fields = ('type', 'instruction')
    ordering = ('-created_at',)


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


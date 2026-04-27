from django.contrib import admin
from .models import Document, ChatSession, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display   = ['name', 'user', 'chunk_count', 'uploaded_at']
    readonly_fields = ['id', 'chunk_count', 'uploaded_at']

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display   = ['title', 'user', 'created_at']
    readonly_fields = ['id', 'created_at']

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display   = ['question', 'session', 'created_at']
    readonly_fields = ['question', 'answer', 'sources', 'created_at']
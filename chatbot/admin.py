from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Document, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display  = ['name', 'chunk_count', 'uploaded_at']
    readonly_fields = ['id', 'chunk_count', 'uploaded_at']

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display  = ['question', 'document', 'created_at']
    readonly_fields = ['question', 'answer', 'sources', 'created_at']
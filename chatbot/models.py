from django.db import models
from django.contrib.auth.models import User
import uuid


class Document(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')  # ← new
    name       = models.CharField(max_length=255)
    file       = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    chunk_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} — {self.name}"


class ChatSession(models.Model):
    """A conversation that can span multiple documents"""
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    documents  = models.ManyToManyField(Document, related_name='sessions')  # ← multi-doc
    title      = models.CharField(max_length=255, default='New Chat')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} — {self.title}"


class ChatMessage(models.Model):
    session    = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    question   = models.TextField()
    answer     = models.TextField()
    sources    = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:50]}"
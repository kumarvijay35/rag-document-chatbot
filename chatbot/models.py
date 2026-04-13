from django.db import models
import uuid

# Create your models here.


class Document(models.Model):
    """Stores uploaded document metadata"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    chunk_count = models.IntegerField(default=0)  # How many chunks were stored
    
    def __str__(self):
        return self.name


class ChatMessage(models.Model):
    """Stores conversation history"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='messages')
    question = models.TextField()
    answer = models.TextField()
    sources = models.JSONField(default=list)  # Store source chunks
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Q: {self.question[:50]}"
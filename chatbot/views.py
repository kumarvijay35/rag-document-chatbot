import os
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken

from .models import Document, ChatSession, ChatMessage
from .rag_service import ingest_document, answer_question, delete_document_vectors


# ── AUTH VIEWS ────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '').strip()
        email    = request.data.get('email', '').strip()

        if not username or not password:
            return Response({"error": "Username and password required"}, status=400)

        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already taken"}, status=400)

        user = User.objects.create_user(username=username, password=password, email=email)
        tokens = RefreshToken.for_user(user)

        return Response({
            "message": "Account created successfully",
            "username": user.username,
            "access":  str(tokens.access_token),
            "refresh": str(tokens),
        }, status=201)


@method_decorator(csrf_exempt, name='dispatch')
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        from django.contrib.auth import authenticate
        username = request.data.get('username', '')
        password = request.data.get('password', '')

        user = authenticate(username=username, password=password)
        if not user:
            return Response({"error": "Invalid credentials"}, status=401)

        tokens = RefreshToken.for_user(user)
        return Response({
            "message": "Login successful",
            "username": user.username,
            "access":  str(tokens.access_token),
            "refresh": str(tokens),
        })


# ── DOCUMENT VIEWS ────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser, FormParser]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=400)

        ext = os.path.splitext(file.name)[1].lower()
        if ext not in ['.pdf', '.txt']:
            return Response({"error": "Only PDF and TXT supported"}, status=400)

        # Save doc linked to the logged-in user
        doc = Document.objects.create(
            user=request.user,
            name=file.name,
            file=file
        )

        try:
            chunks = ingest_document(doc.file.path, str(doc.id))
            doc.chunk_count = chunks
            doc.save()
        except Exception as e:
            doc.delete()
            return Response({"error": str(e)}, status=500)

        return Response({
            "id":      str(doc.id),
            "name":    doc.name,
            "chunks":  doc.chunk_count,
            "message": "Document uploaded and indexed"
        }, status=201)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Only return THIS user's documents
        docs = Document.objects.filter(user=request.user).order_by('-uploaded_at')
        return Response([
            {
                "id":          str(d.id),
                "name":        d.name,
                "chunks":      d.chunk_count,
                "uploaded_at": d.uploaded_at
            }
            for d in docs
        ])


@method_decorator(csrf_exempt, name='dispatch')
class DocumentDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, document_id):
        try:
            doc = Document.objects.get(id=document_id, user=request.user)
        except Document.DoesNotExist:
            return Response({"error": "Not found"}, status=404)

        delete_document_vectors(str(doc.id))
        if doc.file and os.path.exists(doc.file.path):
            os.remove(doc.file.path)
        doc.delete()
        return Response({"message": "Document deleted"})


# ── CHAT SESSION VIEWS ────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class CreateSessionView(APIView):
    """
    POST /api/sessions/
    Body: { "document_ids": ["uuid1", "uuid2"], "title": "My Chat" }
    Creates a chat session linked to selected documents.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        doc_ids = request.data.get('document_ids', [])
        title   = request.data.get('title', 'New Chat')

        if not doc_ids:
            return Response({"error": "Select at least one document"}, status=400)

        # Verify all docs belong to this user
        docs = Document.objects.filter(id__in=doc_ids, user=request.user)
        if docs.count() != len(doc_ids):
            return Response({"error": "One or more documents not found"}, status=404)

        session = ChatSession.objects.create(user=request.user, title=title)
        session.documents.set(docs)

        return Response({
            "session_id":  str(session.id),
            "title":       session.title,
            "documents":   [{"id": str(d.id), "name": d.name} for d in docs],
            "created_at":  session.created_at
        }, status=201)


@method_decorator(csrf_exempt, name='dispatch')
class SessionListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
        return Response([
            {
                "session_id": str(s.id),
                "title":      s.title,
                "documents":  [{"id": str(d.id), "name": d.name} for d in s.documents.all()],
                "created_at": s.created_at
            }
            for s in sessions
        ])


@method_decorator(csrf_exempt, name='dispatch')
class AskQuestionView(APIView):
    """
    POST /api/ask/
    Body: { "session_id": "...", "question": "..." }
    Searches across all documents in the session.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        session_id = request.data.get('session_id')
        question   = request.data.get('question', '').strip()

        if not session_id or not question:
            return Response({"error": "session_id and question required"}, status=400)

        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=404)

        # Get all document collection names for this session
        collection_names = [str(d.id) for d in session.documents.all()]

        try:
            result = answer_question(question, collection_names)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

        ChatMessage.objects.create(
            session=session,
            question=question,
            answer=result["answer"],
            sources=result["sources"]
        )

        return Response({
            "question": question,
            "answer":   result["answer"],
            "sources":  result["sources"]
        })


@method_decorator(csrf_exempt, name='dispatch')
class ChatHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=404)

        messages = session.messages.all().order_by('created_at')
        return Response([
            {
                "question":   m.question,
                "answer":     m.answer,
                "sources":    m.sources,
                "created_at": m.created_at
            }
            for m in messages
        ])
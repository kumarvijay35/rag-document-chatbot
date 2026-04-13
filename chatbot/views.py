import os
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import Document, ChatMessage
from .rag_service import ingest_document, answer_question, delete_document_vectors


@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        file = request.FILES.get('file')

        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        allowed_extensions = ['.pdf', '.txt']
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_extensions:
            return Response({"error": "Only PDF and TXT files are supported"}, status=status.HTTP_400_BAD_REQUEST)

        doc = Document.objects.create(name=file.name, file=file)

        try:
            chunk_count = ingest_document(doc.file.path, collection_name=str(doc.id))
            doc.chunk_count = chunk_count
            doc.save()
        except Exception as e:
            doc.delete()
            return Response({"error": f"Failed to process document: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "id": str(doc.id),
            "name": doc.name,
            "chunks": chunk_count,
            "message": "Document uploaded and indexed successfully"
        }, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentListView(APIView):

    def get(self, request):
        docs = Document.objects.all().order_by('-uploaded_at')
        data = [
            {
                "id": str(d.id),
                "name": d.name,
                "chunks": d.chunk_count,
                "uploaded_at": d.uploaded_at
            }
            for d in docs
        ]
        return Response(data)


@method_decorator(csrf_exempt, name='dispatch')
class AskQuestionView(APIView):

    def post(self, request):
        document_id = request.data.get('document_id')
        question = request.data.get('question', '').strip()

        if not document_id or not question:
            return Response({"error": "Both document_id and question are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            doc = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            result = answer_question(question, collection_name=str(doc.id))
        except Exception as e:
            return Response({"error": f"Failed to get answer: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        ChatMessage.objects.create(
            document=doc,
            question=question,
            answer=result["answer"],
            sources=result["sources"]
        )

        return Response({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })


@method_decorator(csrf_exempt, name='dispatch')
class ChatHistoryView(APIView):

    def get(self, request, document_id):
        try:
            doc = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=404)

        messages = doc.messages.all().order_by('created_at')
        data = [
            {
                "question": m.question,
                "answer": m.answer,
                "sources": m.sources,
                "created_at": m.created_at
            }
            for m in messages
        ]
        return Response(data)


@method_decorator(csrf_exempt, name='dispatch')
class DocumentDeleteView(APIView):

    def delete(self, request, document_id):
        try:
            doc = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=404)

        delete_document_vectors(collection_name=str(doc.id))

        if doc.file and os.path.exists(doc.file.path):
            os.remove(doc.file.path)

        doc.delete()
        return Response({"message": "Document deleted successfully."})
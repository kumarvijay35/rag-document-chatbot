from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.DocumentUploadView.as_view(), name='upload'),
    path('documents/', views.DocumentListView.as_view(), name='documents'),
    path('ask/', views.AskQuestionView.as_view(), name='ask'),
    path('history/<str:document_id>/', views.ChatHistoryView.as_view(), name='history'),
    path('documents/<str:document_id>/', views.DocumentDeleteView.as_view(), name='delete-document'),
]

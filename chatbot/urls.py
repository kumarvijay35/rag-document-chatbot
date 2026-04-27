from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('auth/register/', views.RegisterView.as_view(),    name='register'),
    path('auth/login/',    views.LoginView.as_view(),       name='login'),

    # Documents
    path('upload/',                        views.DocumentUploadView.as_view(), name='upload'),
    path('documents/',                     views.DocumentListView.as_view(),   name='documents'),
    path('documents/<str:document_id>/',   views.DocumentDeleteView.as_view(), name='delete-doc'),

    # Chat sessions
    path('sessions/',                      views.CreateSessionView.as_view(),  name='create-session'),
    path('sessions/list/',                 views.SessionListView.as_view(),    name='list-sessions'),
    path('ask/',                           views.AskQuestionView.as_view(),    name='ask'),
    path('history/<str:session_id>/',      views.ChatHistoryView.as_view(),    name='history'),
]
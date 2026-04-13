# RAG Document Chatbot 🤖📄

An AI-powered document Q&A system. Upload any PDF or TXT and ask questions about it in plain English.

## 🚀 Live Demo

Upload a document → Select it → Ask questions → Get AI answers with source references.

## ✨ Features

- Upload PDF or TXT documents
- Documents chunked, embedded, stored in ChromaDB vector database
- Ask questions in natural language
- AI answers using ONLY content from your document (no hallucination)
- Full chat history saved per document
- Clean responsive frontend UI
- Django Admin panel

## 🛠️ Tech Stack

| Layer           | Technology                                 |
| --------------- | ------------------------------------------ |
| Backend         | Django 5.2, Django REST Framework          |
| AI Pipeline     | LangChain, RetrievalQA Chain               |
| Embeddings      | HuggingFace all-MiniLM-L6-v2 (free, local) |
| Vector Database | ChromaDB (local)                           |
| LLM             | Groq API — Llama 3.3 70B (free tier)       |
| Frontend        | HTML, CSS, Vanilla JavaScript              |

## ⚙️ How RAG Works

## 🔧 Setup

### 1. Clone & install

```bash
git clone https://github.com/kumarvijay35/rag-document-chatbot.git
cd rag-document-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

GROQ_API_KEY= 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

Get a free key at https://console.groq.com

### 3. Run

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000`

## 📡 API Endpoints

| Method | Endpoint               | Description        |
| ------ | ---------------------- | ------------------ |
| POST   | `/api/upload/`         | Upload a document  |
| GET    | `/api/documents/`      | List all documents |
| POST   | `/api/ask/`            | Ask a question     |
| GET    | `/api/history/<id>/`   | Get chat history   |
| DELETE | `/api/documents/<id>/` | Delete a document  |

## 📁 Project Structure

rag_project/
├── chatbot/
│ ├── models.py # Document & ChatMessage models
│ ├── views.py # REST API endpoints
│ ├── rag_service.py # RAG pipeline (core AI logic)
│ └── urls.py
├── core/
│ ├── settings.py
│ └── urls.py
├── templates/
│ └── index.html # Frontend UI
└── requirements.txt

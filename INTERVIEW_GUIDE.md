# RAG Document Chatbot — Complete Code Walkthrough & Interview Guide

A study guide for explaining this project line-by-line and defending it in an interview.

---

## 1. The 30-Second Pitch (memorize this)

> "It's a full-stack **RAG (Retrieval-Augmented Generation)** document Q&A app. A user logs in, uploads a PDF or TXT, and asks questions in plain English. Behind the scenes I split the document into chunks, convert each chunk into a vector embedding, and store them in a ChromaDB vector database. When a question comes in, I embed the question, retrieve the most similar chunks, and feed only those chunks to an LLM (Llama 3.3 70B via Groq) with a strict prompt that says *answer only from this context*. That grounds the model in the user's document and prevents hallucination. The backend is Django REST Framework with JWT auth; the frontend is plain HTML/CSS/JS."

**Why RAG instead of just an LLM?** An LLM doesn't know your private document and will hallucinate. RAG injects the *relevant* portion of your document into the prompt at query time, so answers are grounded, current, and cite their source — without retraining the model.

---

## 2. Architecture / Data Flow

```
                         ┌─────────────── BROWSER (index.html) ───────────────┐
                         │  Login → Upload doc → Select docs → Ask question   │
                         └───────────────┬────────────────────────────────────┘
                                         │ JWT in Authorization header
                                         ▼
                ┌──────────────── DJANGO REST API (views.py) ────────────────┐
                │  /api/auth/   /api/upload/   /api/sessions/   /api/ask/     │
                └───────┬───────────────────────────┬─────────────────────────┘
                        │ (ingest)                  │ (query)
                        ▼                            ▼
            ┌────────── rag_service.py (the AI core) ──────────┐
            │  INGEST: load → split → embed → store in Chroma  │
            │  QUERY:  embed question → retrieve top-k chunks  │
            │          → stuff into prompt → Groq LLM → answer │
            └───────┬──────────────────────────┬──────────────┘
                    ▼                           ▼
            ChromaDB (vectors)          Groq API (Llama 3.3 70B)
                    │
            SQLite (metadata: users, documents, sessions, messages)
```

**Two pipelines to name in an interview:**
1. **Ingestion pipeline** (on upload): `load_document → split_documents → embed → Chroma.from_documents`
2. **Query pipeline** (on ask): `embed question → retriever (top-k) → RetrievalQA "stuff" chain → LLM → answer + sources`

---

## 3. Tech Stack & Why Each Choice

| Layer | Choice | Why (interview answer) |
|---|---|---|
| Web framework | Django 5.2 + DRF | Batteries-included: ORM, auth, admin, migrations out of the box |
| Auth | JWT (simplejwt) | Stateless tokens — no server-side session store; scales horizontally |
| AI orchestration | LangChain | Standard abstractions for loaders, splitters, retrievers, chains |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, runs locally on CPU, 384-dim, fast, no API cost |
| Vector DB | ChromaDB | Lightweight, embeddable, persists to disk, no separate server |
| LLM | Groq — Llama 3.3 70B | Free tier, extremely fast inference, strong open model |
| DB | SQLite | Zero-config for a demo; swap to Postgres for production |
| Frontend | Vanilla HTML/CSS/JS | No build step; keeps focus on the backend/AI |

---

## 4. Project Structure

```
rag_project/
├── core/                 # Django project config
│   ├── settings.py       # apps, JWT, CORS, DB, static
│   └── urls.py           # top-level routes (admin, /api/, home page)
├── chatbot/              # the app
│   ├── models.py         # Document, ChatSession, ChatMessage tables
│   ├── rag_service.py    # ← THE AI CORE: ingest + query pipelines
│   ├── views.py          # REST API endpoints
│   ├── urls.py           # /api/ route map
│   ├── admin.py          # Django admin registration
│   └── migrations/       # DB schema as code
├── templates/index.html  # single-page frontend
├── manage.py             # Django CLI entrypoint
├── requirements.txt
└── .env                  # GROQ_API_KEY (kept out of git)
```

---

## 5. Line-by-Line: `chatbot/rag_service.py` (THE MOST IMPORTANT FILE)

This is the file the interviewer will dig into. Know it cold.

```python
import os
from dotenv import load_dotenv
```
Standard library `os` for file paths/env vars; `load_dotenv` reads the `.env` file so the Groq key isn't hard-coded.

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever
```
The whole RAG toolkit:
- **PyPDFLoader / TextLoader** — turn a file into LangChain `Document` objects (text + metadata).
- **RecursiveCharacterTextSplitter** — break long text into overlapping chunks.
- **HuggingFaceEmbeddings** — the model that turns text into vectors.
- **Chroma** — the vector store wrapper.
- **ChatGroq** — the LLM client.
- **RetrievalQA** — the chain that wires retriever + LLM together.
- **PromptTemplate** — the instruction wrapper that enforces "answer only from context."
- **MergerRetriever** — lets one question search across *multiple* documents at once.

```python
load_dotenv()
CHROMA_DB_DIR = "/tmp/chroma_db"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
```
Load env vars, set where ChromaDB persists, read the API key.
*(Interview note: `/tmp` is fine for a free-hosting demo but ephemeral — on Render/Heroku `/tmp` is wiped on restart. For production you'd use a persistent volume or a managed vector DB.)*

### `get_embeddings()`
```python
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
```
Returns the embedding model. `all-MiniLM-L6-v2` maps text → **384-dimensional** vectors. `device="cpu"` (no GPU needed). `normalize_embeddings=True` makes vectors unit-length so **cosine similarity = dot product**, which makes nearest-neighbor search consistent.

### `load_document()`
```python
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    if not documents:
        raise ValueError("Document is empty.")
    return documents
```
Picks the right loader by file extension, loads the raw text, and guards against empty files. Returns a list of `Document` objects (one per PDF page for PDFs).

### `split_documents()` — chunking
```python
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)
```
**The single most-asked design question: "Why chunk, and why these numbers?"**
- LLMs and embeddings have a context limit; you can't embed a whole book as one vector and retrieval would be useless if you did.
- `chunk_size=600` chars ≈ a paragraph — small enough to be a precise retrieval unit, big enough to hold a complete thought.
- `chunk_overlap=100` repeats the last 100 chars of one chunk at the start of the next so a sentence split across a boundary isn't lost.
- `separators` = "try to split on paragraph breaks first, then line breaks, then sentences, then words" — keeps chunks semantically clean instead of cutting mid-word.

### `ingest_document()` — the WRITE path
```python
def ingest_document(file_path: str, collection_name: str) -> int:
    documents = load_document(file_path)
    chunks    = split_documents(documents)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    return len(chunks)
```
Load → split → embed every chunk → store vectors in ChromaDB. **Key design point:** `collection_name` is the document's UUID, so **each document gets its own isolated collection.** That's how the app keeps one user's docs separate and how it can search a specific subset. Returns the chunk count (shown in the UI).

### `PROMPT_TEMPLATE` — the anti-hallucination guardrail
```python
PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I could not find this in the selected documents."
Do NOT make up information.
Context:
{context}
Question: {question}
Answer:
"""
```
This is **prompt engineering doing the safety work.** `{context}` gets filled with retrieved chunks, `{question}` with the user's question. The explicit "ONLY the context" + "say you couldn't find it" instruction is what keeps the model grounded. Worth calling out as a deliberate design decision.

### `answer_question()` — the READ path (the heart of query-time RAG)
```python
def answer_question(question: str, collection_names: list) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not collection_names:
        raise ValueError("No documents selected.")
    embeddings = get_embeddings()
```
Input validation, then load the **same** embedding model used at ingest — critical, because question and chunks must live in the same vector space to be comparable.

```python
    retrievers = []
    for col in collection_names:
        try:
            vs = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=col
            )
            retrievers.append(
                vs.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            )
        except Exception as e:
            print(f"[Query] Skipping collection {col}: {e}")
```
For each selected document, open its Chroma collection and build a retriever that returns the **top `k=2`** most similar chunks. The `try/except` skips any missing/corrupt collection instead of crashing the whole query.

```python
    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        retriever = MergerRetriever(retrievers=retrievers)
```
One document → use its retriever directly. Multiple documents → `MergerRetriever` runs all of them and merges the results, enabling **multi-document Q&A in one question.**

```python
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY
    )
```
The LLM. **`temperature=0`** = deterministic, factual output (no creative randomness) — exactly what you want for document Q&A. `max_tokens=1024` caps answer length.

```python
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
```
Wire it all together. **`chain_type="stuff"`** = take all retrieved chunks and "stuff" them straight into the one prompt. (Be ready to name the alternatives: `map_reduce`, `refine`, `map_rerank` — used when you retrieve too many chunks to fit in one prompt. "Stuff" is the right call here because `k=2` keeps context small.) `return_source_documents=True` so we can show the user *where* the answer came from.

```python
    result  = qa_chain.invoke({"query": question})
    answer  = result.get("result", "No answer returned.")
    sources = [doc.page_content[:300] for doc in result.get("source_documents", [])]
    return {"answer": answer, "sources": sources}
```
Run the chain, pull out the answer text, and grab the first 300 chars of each source chunk for citation. Return both — citations build user trust and prove the answer is grounded.

### `delete_document_vectors()`
```python
def delete_document_vectors(collection_name: str):
    try:
        vs = Chroma(persist_directory=CHROMA_DB_DIR,
                    embedding_function=get_embeddings(),
                    collection_name=collection_name)
        vs.delete_collection()
    except Exception as e:
        print(f"[Delete] {e}")
```
When a document is deleted, drop its entire vector collection so no orphaned embeddings remain. Good "data hygiene" point.

---

## 6. `chatbot/models.py` — the data model

```python
class Document(models.Model):
    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    name        = models.CharField(max_length=255)
    file        = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    chunk_count = models.IntegerField(default=0)
```
- **UUID primary key** (not auto-increment 1,2,3): unguessable, safe to expose in URLs, and globally unique — and it doubles as the Chroma collection name.
- **`user` ForeignKey + `on_delete=CASCADE`**: every doc belongs to a user; delete the user → their docs go too. This is the basis of **data isolation** (each user only sees their own data).
- `FileField(upload_to='documents/')` saves the upload under `media/documents/`.
- `chunk_count` cached so the UI can show it without re-reading the file.

```python
class ChatSession(models.Model):
    id        = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user      = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    documents = models.ManyToManyField(Document, related_name='sessions')
    title     = models.CharField(max_length=255, default='New Chat')
    created_at = models.DateTimeField(auto_now_add=True)
```
A conversation. **`ManyToManyField` to Document** is the key design choice — one session can span several documents, and a document can appear in many sessions. That's what powers multi-doc chat.

```python
class ChatMessage(models.Model):
    session    = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    question   = models.TextField()
    answer     = models.TextField()
    sources    = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
```
One Q&A turn. `sources` is a `JSONField` (a list of source snippets) so the citations persist with the message. `related_name='messages'` lets you do `session.messages.all()` to rebuild chat history.

**One-liner to remember:** *User → has many Documents and Sessions; Session ↔ many Documents (M2M); Session → has many Messages.*

---

## 7. `chatbot/views.py` — the REST API

Pattern: every endpoint is a DRF `APIView`, almost all require `IsAuthenticated`, and every query is **scoped to `request.user`** so users can't touch each other's data.

- **`RegisterView`** (`AllowAny`) — validates username/password, rejects duplicates, calls `User.objects.create_user` (which **hashes** the password — never stored plaintext), and returns a JWT access+refresh pair so the user is logged in immediately.
- **`LoginView`** (`AllowAny`) — `authenticate()` checks credentials; on success issues JWTs; on failure returns 401.
- **`DocumentUploadView`** — checks a file was sent, validates the extension is `.pdf`/`.txt`, creates the `Document` row, then calls `ingest_document(...)`. **Note the rollback:** if ingestion throws, it `doc.delete()`s the half-created record so you never have a document row without vectors. Returns chunk count.
- **`DocumentListView`** — `Document.objects.filter(user=request.user)` → only your docs.
- **`DocumentDeleteView`** — fetches the doc *scoped to the user* (so you can't delete someone else's by guessing the ID), deletes the vectors, deletes the file from disk, deletes the row.
- **`CreateSessionView`** — validates that **all** requested document IDs actually belong to the user (`docs.count() != len(doc_ids)` guard) before creating the session and linking docs via `.set(docs)`.
- **`AskQuestionView`** — the orchestrator: load the session (user-scoped), gather its documents' collection names, call `answer_question(...)`, **persist the Q&A as a `ChatMessage`**, and return the answer + sources.
- **`ChatHistoryView`** — returns all messages for a session, oldest first, to rebuild the chat on reload.

**`@method_decorator(csrf_exempt)`**: CSRF protection is for cookie-based sessions; this API uses **JWT bearer tokens** (not cookies), so CSRF doesn't apply and is disabled. Be ready to explain that — it's a common interview probe.

---

## 8. `chatbot/urls.py` & `core/urls.py` — routing

`core/urls.py` mounts `admin/`, includes the app's routes under `api/`, serves `index.html` at `/`, and serves uploaded media in dev. `chatbot/urls.py` maps each path to its view: `auth/register/`, `auth/login/`, `upload/`, `documents/`, `documents/<id>/`, `sessions/`, `sessions/list/`, `ask/`, `history/<id>/`. Clean, RESTful, resource-oriented.

---

## 9. `core/settings.py` — config highlights

- `INSTALLED_APPS` adds `rest_framework`, `rest_framework_simplejwt`, `corsheaders`, and the `chatbot` app.
- `REST_FRAMEWORK` sets **JWT as the default auth** and `IsAuthenticated` as the **default permission** (secure-by-default — endpoints are locked unless explicitly opened).
- `SIMPLE_JWT`: access token 24h, refresh token 7 days, `Bearer` header type.
- `CORS_ALLOW_ALL_ORIGINS = True` lets the browser frontend call the API cross-origin.
- `WhiteNoise` serves static files in production without needing nginx/S3.
- SQLite database; `MEDIA_ROOT` for uploads.

**Things you should flag yourself before the interviewer does (shows maturity):**
- `SECRET_KEY` is hard-coded and `DEBUG=True` — fine for a demo, **must** be env-var + `DEBUG=False` in production.
- `CORS_ALLOW_ALL_ORIGINS=True` should be a specific allowlist in production.
- The `.env` with a real `GROQ_API_KEY` should be rotated and never committed (it *is* in `.gitignore`).

---

## 10. `templates/index.html` — the frontend (what to say)

Single-page app, no framework. Key points:
- **State**: JWT `token` and username kept in `localStorage` so a refresh keeps you logged in (`if (token) showApp()` at the bottom).
- **`authHeaders()`** attaches `Authorization: Bearer <token>` to every API call.
- **Flow**: `login()/register()` → store token → `showApp()` → `loadDocs()` + `loadSessions()`.
- **Upload** uses `FormData` (multipart) because it's a file; everything else is JSON.
- **`selectedDocIds` (a `Set`)** tracks which docs are checked → sent to `CreateSessionView`.
- **`askQuestion()`** shows a "⏳ Searching documents..." placeholder, POSTs to `/ask/`, then renders the answer with its **Sources** section.
- Honest caveat: `localStorage` for JWTs is convenient but XSS-vulnerable; `httpOnly` cookies are safer for production.

---

## 11. Likely Interview Questions + Crisp Answers

**Q: What is RAG and why use it?**
Retrieval-Augmented Generation. You retrieve relevant context from your own data and inject it into the LLM prompt at query time. It grounds answers in private/current data and cuts hallucination without retraining the model.

**Q: What's an embedding?**
A numeric vector that captures the *meaning* of text. Semantically similar texts get nearby vectors, so "find relevant chunks" becomes "find nearest vectors."

**Q: How does retrieval actually find the right chunks?**
The question is embedded with the same model, then ChromaDB does an approximate nearest-neighbor search (HNSW index) over the stored chunk vectors and returns the top-k by cosine similarity.

**Q: Why chunk overlap?**
So a sentence or idea split across a chunk boundary still appears intact in at least one chunk — prevents losing context at the seams.

**Q: How do you prevent hallucination?**
Three layers: (1) only feed retrieved context, (2) `temperature=0`, (3) an explicit prompt instruction to answer only from context and admit when the answer isn't there.

**Q: Why `temperature=0`?**
Deterministic, factual answers. Higher temperature adds randomness/creativity, which is wrong for document Q&A.

**Q: What does `chain_type="stuff"` mean and what are the alternatives?**
"Stuff" puts all retrieved chunks into one prompt. Alternatives: `map_reduce` (summarize each chunk then combine), `refine` (iteratively improve), `map_rerank` (score and pick best). Stuff works here because k is small.

**Q: How do you isolate one user's data from another's?**
Every model has a `user` ForeignKey, and every query is filtered by `request.user`. JWT identifies the user on each request. Each document's vectors live in their own UUID-named Chroma collection.

**Q: Why JWT over Django sessions?**
Stateless — no server-side session store, scales horizontally, and works cleanly for an API consumed by a separate frontend or mobile client.

**Q: How would you scale this?**
Swap SQLite → Postgres; move Chroma → a managed/server vector DB (or pgvector); run ingestion as a **background job** (Celery) instead of blocking the upload request; cache the embedding model in memory; add pagination and rate limiting; persistent storage for vectors (not `/tmp`).

**Q: Biggest weakness of the current code?**
Ingestion is synchronous — a large PDF blocks the HTTP request and could time out. Fix: process it in a Celery worker and have the UI poll for "indexed" status. Also the embedding model is reloaded on every call — should be a cached singleton.

**Q: What if the answer spans two documents?**
`MergerRetriever` pulls top chunks from each selected document's collection and merges them, so the LLM sees context from all of them at once.

---

## 12. Known Limitations (say these *before* they're asked — it reads as senior)

1. Synchronous ingestion (should be async/Celery).
2. `get_embeddings()` reloads the model each call (should be a cached singleton).
3. Vectors persist to `/tmp` — ephemeral on most PaaS hosts.
4. `DEBUG=True`, hard-coded `SECRET_KEY`, `CORS_ALLOW_ALL_ORIGINS` — demo-only settings.
5. No reranking — for larger corpora, add a cross-encoder reranker after retrieval.
6. JWT in `localStorage` is XSS-exposed; prefer `httpOnly` cookies in production.
7. No tests beyond a manual `test_rag.py` script — would add unit/integration tests.

---

## 13. One-Paragraph Summary (fallback if you only have 20 seconds)

> "Django REST backend with JWT auth, plus a LangChain RAG pipeline. On upload I chunk the document, embed each chunk with a local HuggingFace MiniLM model, and store the vectors in ChromaDB under a per-document collection. On a question, I embed the query, retrieve the top-2 similar chunks per selected document, merge them, and pass them to Llama 3.3 70B on Groq with a strict 'answer only from context' prompt at temperature 0. The answer plus source snippets get saved as a chat message and returned to a vanilla-JS frontend."

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

CHROMA_DB_DIR = "chroma_db"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────
# EMBEDDINGS  (runs locally, free, no API)
# ─────────────────────────────────────────

def get_embeddings():
    """
    HuggingFace sentence-transformers model.
    Downloads once (~90 MB), then runs fully offline.
    'all-MiniLM-L6-v2' is small, fast, and accurate enough for RAG.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

# ─────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────

def load_document(file_path: str):
    """
    Loads a PDF or TXT file and returns a list of LangChain Document objects.
    Each Document has .page_content (text) and .metadata (source, page number).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Only .pdf and .txt are supported."
        )

    documents = loader.load()

    if not documents:
        raise ValueError("The document appears to be empty or could not be read.")

    return documents

# ─────────────────────────────────────────
# TEXT SPLITTING
# ─────────────────────────────────────────

def split_documents(documents):
    """
    Splits documents into overlapping chunks.

    Why split?  LLMs have a token limit. A 100-page PDF can't fit in one prompt.
    We split it into small pieces, store them all, then only retrieve the
    relevant ones when a question is asked.

    chunk_size=600    → each chunk is ~600 characters
    chunk_overlap=100 → neighbouring chunks share 100 chars so context isn't lost
                        at the boundary between two chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks. File may be empty.")

    return chunks

# ─────────────────────────────────────────
# INGESTION PIPELINE
# ─────────────────────────────────────────

def ingest_document(file_path: str, collection_name: str) -> int:
    """
    Full ingestion pipeline — call this when a user uploads a document.

    Steps:
      1. Load the file
      2. Split into chunks
      3. Embed each chunk using HuggingFace
      4. Store embeddings in ChromaDB under the given collection_name

    Returns the number of chunks stored.
    """
    print(f"[Ingest] Loading file: {file_path}")
    documents = load_document(file_path)
    print(f"[Ingest] Loaded {len(documents)} page(s)")

    chunks = split_documents(documents)
    print(f"[Ingest] Split into {len(chunks)} chunks")

    print("[Ingest] Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    vectorstore.persist()
    print(f"[Ingest] Done. {len(chunks)} chunks stored in collection '{collection_name}'")

    return len(chunks)

# ─────────────────────────────────────────
# CUSTOM PROMPT TEMPLATE
# ─────────────────────────────────────────

PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I could not find the answer in the document."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:
"""

def get_prompt():
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ─────────────────────────────────────────
# QUERY PIPELINE
# ─────────────────────────────────────────

def answer_question(question: str, collection_name: str) -> dict:
    """
    Full query pipeline — call this when a user asks a question.

    Steps:
      1. Load the ChromaDB collection for this document
      2. Embed the question and find the top-3 most similar chunks
      3. Send those chunks + the question to Groq's Llama 3
      4. Return the answer and the source chunks used

    Args:
        question:        The user's question string
        collection_name: The UUID of the document (used as ChromaDB collection)

    Returns:
        {
          "answer":  "The AI's answer...",
          "sources": ["chunk text 1...", "chunk text 2...", ...]
        }
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Load vectorstore
    print(f"[Query] Loading collection '{collection_name}' from ChromaDB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection_name
    )

    # Step 2: Create retriever — finds top 3 relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Step 3: Load Groq LLM
    print("[Query] Initialising Groq LLM...")
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,            # 0 = factual, deterministic answers
        max_tokens=1024,          # max length of the answer
        groq_api_key=GROQ_API_KEY
    )

    # Step 4: Build RetrievalQA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # "stuff" = put all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt()}
    )

    # Step 5: Run the chain
    print(f"[Query] Running chain for question: '{question}'")
    result = qa_chain.invoke({"query": question})

    answer  = result.get("result", "No answer returned.")
    sources = [
        doc.page_content[:300]           # first 300 chars of each source chunk
        for doc in result.get("source_documents", [])
    ]

    print(f"[Query] Answer: {answer[:100]}...")
    return {
        "answer":  answer,
        "sources": sources
    }

# ─────────────────────────────────────────
# UTILITY — delete a document's vectors
# ─────────────────────────────────────────

def delete_document_vectors(collection_name: str):
    """
    Deletes all stored vectors for a document from ChromaDB.
    Call this when a user deletes an uploaded document.
    """
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings(),
            collection_name=collection_name
        )
        vectorstore.delete_collection()
        print(f"[Delete] Collection '{collection_name}' deleted from ChromaDB.")
    except Exception as e:
        print(f"[Delete] Could not delete collection: {e}")
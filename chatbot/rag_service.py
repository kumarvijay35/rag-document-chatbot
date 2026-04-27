import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever   # ← new

load_dotenv()

CHROMA_DB_DIR = "/tmp/chroma_db"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


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


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def ingest_document(file_path: str, collection_name: str) -> int:
    print(f"[Ingest] Loading: {file_path}")
    documents = load_document(file_path)
    chunks    = split_documents(documents)
    print(f"[Ingest] {len(chunks)} chunks — embedding...")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    return len(chunks)


PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I could not find this in the selected documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:
"""


def answer_question(question: str, collection_names: list) -> dict:
    """
    Multi-document RAG query.
    collection_names: list of document UUIDs to search across.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not collection_names:
        raise ValueError("No documents selected.")

    embeddings = get_embeddings()

    # Build one retriever per document, then merge them
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

    if not retrievers:
        raise ValueError("No valid document collections found.")

    # MergerRetriever combines results from all retrievers
    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        retriever = MergerRetriever(retrievers=retrievers)

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY
    )

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

    result  = qa_chain.invoke({"query": question})
    answer  = result.get("result", "No answer returned.")
    sources = [doc.page_content[:300] for doc in result.get("source_documents", [])]

    return {"answer": answer, "sources": sources}


def delete_document_vectors(collection_name: str):
    try:
        vs = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings(),
            collection_name=collection_name
        )
        vs.delete_collection()
    except Exception as e:
        print(f"[Delete] {e}")
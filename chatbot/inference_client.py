"""
chatbot/inference_client.py

The ONLY place Django talks to the FastAPI inference service.
Matches the actual RAG project design: chat sessions span multiple
documents, so query takes a list of document ids.

settings.py additions:
    INFERENCE_SERVICE_URL = os.environ.get("INFERENCE_SERVICE_URL", "http://localhost:8001")
    INFERENCE_INTERNAL_API_KEY = os.environ.get("INFERENCE_INTERNAL_API_KEY")

requirements: httpx
"""

import httpx
from django.conf import settings

# Explicit timeouts — an unbounded HTTP call from a sync Django worker is how
# you take down the whole app when a downstream service hangs.
TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class InferenceServiceError(Exception):
    """Raised when the inference service is unreachable or returns an error."""


def _headers() -> dict:
    return {"X-Internal-Api-Key": settings.INFERENCE_INTERNAL_API_KEY}


def _post(path: str, payload: dict) -> dict:
    try:
        resp = httpx.post(
            f"{settings.INFERENCE_SERVICE_URL}{path}",
            json=payload,
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        raise InferenceServiceError(f"{path} failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise InferenceServiceError(f"{path} unreachable: {exc}") from exc


def embed_document(user_id, document_id, text: str, filename: str = "") -> dict:
    """Index extracted document text. Returns {"chunks_indexed": int, ...}."""
    return _post(
        "/v1/embed",
        {
            "user_id": str(user_id),
            "document_id": str(document_id),
            "text": text,
            "metadata": {"filename": filename} if filename else None,
        },
    )


def query_documents(user_id, question: str, document_ids: list, top_k: int = 3) -> dict:
    """Ask a question across a session's documents.

    Returns {"answer": str, "sources": [{"text", "document_id", "score"}], "model": str}
    """
    return _post(
        "/v1/query",
        {
            "user_id": str(user_id),
            "question": question,
            "document_ids": [str(d) for d in document_ids],
            "top_k": top_k,
        },
    )


def delete_document(user_id, document_id) -> None:
    try:
        resp = httpx.delete(
            f"{settings.INFERENCE_SERVICE_URL}/v1/documents/{user_id}/{document_id}",
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise InferenceServiceError(f"delete failed: {exc}") from exc

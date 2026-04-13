from chatbot.rag_service import ingest_document, answer_question
import django, os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

# 1. Create a simple test text file
with open("test_doc.txt", "w") as f:
    f.write("""
    Sachin Tendulkar is an Indian cricketer widely regarded as one of the greatest
    batsmen of all time. He scored 100 international centuries during his career.
    He retired from all forms of cricket in 2013. He was born on April 24, 1973
    in Mumbai, India. He is also known as the 'Master Blaster'.
    """)

# 2. Ingest it
print("Ingesting document...")
chunks = ingest_document("test_doc.txt", collection_name="test-collection-001")
print(f"Stored {chunks} chunks\n")

# 3. Ask a question
print("Asking question...")
result = answer_question(
    question="How many centuries did Sachin Tendulkar score?",
    collection_name="test-collection-001"
)

print("\n=== ANSWER ===")
print(result["answer"])
print("\n=== SOURCES USED ===")
for i, src in enumerate(result["sources"], 1):
    print(f"[{i}] {src}")
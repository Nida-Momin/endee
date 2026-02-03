from embeddings.embedding_store import EmbeddingStore
from rag.rag_pipeline import RAGPipeline

documents = [
    "Endee is a vector database",
    "RAG combines retrieval and generation",
    "Semantic search uses embeddings"
]

query = "What is RAG?"

store = EmbeddingStore()
store.add_documents(documents)

top_docs = store.search(query)
context = " ".join(top_docs)

rag = RAGPipeline()
answer = rag.answer(query, context)

print("Question:", query)
print("Answer:", answer)

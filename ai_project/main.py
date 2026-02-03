from embeddings.embedding_store import EmbeddingStore
from rag.rag_pipeline import RAGPipeline

# Example documents
documents = [
    "Endee is a vector database",
    "RAG combines retrieval and generation",
    "Semantic search uses embeddings"
]

# Example query
query = "What is RAG?"

# Initialize vector store and add documents
store = EmbeddingStore()
store.add_documents(documents)

# Retrieve top documents
top_docs = store.search(query)
context = " ".join(top_docs)

# Initialize RAG pipeline and get answer
rag = RAGPipeline()
answer = rag.answer(query, context)

print("Question:", query)
print("Answer:", answer)

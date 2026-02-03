from sentence_transformers import SentenceTransformer

class EmbeddingStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectors = []

    def add_documents(self, documents):
        for doc in documents:
            vector = self.embedder.encode(doc)
            self.vectors.append((vector, doc))

    def search(self, query, top_k=2):
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.embedder.encode(query)

        scores = []
        for vec, doc in self.vectors:
            score = cosine_similarity([query_vec], [vec])[0][0]
            scores.append((score, doc))

        scores.sort(reverse=True)
        return [doc for _, doc in scores[:top_k]]

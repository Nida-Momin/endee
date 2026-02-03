# RAG-based Semantic Search using Endee

## Overview
This project demonstrates a **Retrieval Augmented Generation (RAG)** pipeline using **Endee** as the vector database framework.  
It implements semantic search using sentence embeddings and uses a transformer-based question-answering model for generating answers from retrieved documents.

## Use Case
- Semantic Search
- Retrieval Augmented Generation (RAG)
- Vector similarity search using embeddings

## Architecture
1. Documents are embedded using **Sentence Transformers**.
2. Embeddings are stored in an **Endee-style vector store**.
3. User queries are embedded and matched using **cosine similarity**.
4. Top-k results are retrieved and passed to a **language model** (FLAN-T5) for answer generation.

## Tech Stack
- Python
- Endee (Vector DB framework)
- Sentence Transformers
- HuggingFace Transformers (FLAN-T5)
- scikit-learn

## Project Structure

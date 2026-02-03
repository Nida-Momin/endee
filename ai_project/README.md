# RAG-based Semantic Search using Endee

## Overview
This project demonstrates a Retrieval Augmented Generation (RAG) pipeline
using Endee as a vector database framework. It implements semantic search
with sentence embeddings and uses a transformer-based QA model for answers.

## Architecture
1. Embed documents with Sentence Transformers.
2. Store embeddings in an Endee-style vector store.
3. Embed user query and retrieve top-k documents via cosine similarity.
4. Pass retrieved context to FLAN-T5 for answer generation.

## Tech Stack
- Python
- Endee (vector database)
- Sentence Transformers
- Hugging Face Transformers (FLAN-T5)
- scikit-learn

## Run
```bash
python ai_project/main.py
```
## Project Structure

ai_project/
├── data/ # Optional data storage
├── embeddings/ # Vector store module
│ └── embedding_store.py
├── rag/ # RAG pipeline module
│ └── rag_pipeline.py
├── notebooks/ # Demo notebooks
├── main.py # Run RAG demo
└── README.md # Project description

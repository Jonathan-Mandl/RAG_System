# Hebrew RAG System - "כל זכות" webiste

This repository contains a Python-based implementation of a Retrieval-Augmented Generation (RAG) system in Hebrew for the "כל זכות" website. The system is designed to enhance the capabilities of language models by integrating external knowledge from website documents through retrieval mechanisms.

## Features

- **Document Indexing (`bm25_indexing.py`)**: Utilizes BM25 for efficient document indexing and retrieval.
- **Text Lemmatization (`lemmatize_text.py`)**: Processes text data to extract lemmatized forms (using `dictabert-lex` model), improving retrieval accuracy with BM25 indexing.
- **Indexable Unit Extraction (`extract_indexable_units.py`)**: Identifies and extracts units suitable for indexing from HTML documents using `Beautiful Soup` and `markdownify`.
- **Utility Functions (`Utils.py`)**: Includes helper functions to support the RAG system's operations.
- **RAG Pipeline (`RAG_system.py`)**: Full implementation of the RAG pipeline using OpenAI's API.
- **Chat App (`RAG_chat.py`)**: Code for an interactive Streamlit application.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jonathan-Mandl/RAG_System.git
   cd RAG_System

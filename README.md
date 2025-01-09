# RAG_System

RAG_System is a Python-based implementation of a Retrieval-Augmented Generation (RAG) system for the "כל זכות" website in herbew.The system is designed to enhance the capabilities of language models by integrating external knowledge from website documents through retrieval mechanisms.

## Features

- **Document Indexing**: Utilizes BM25 for efficient document indexing and retrieval.
- **Text Lemmatization**: Processes text data to extract lemmatized forms (using dictabert-lex model), improving retrieval accuracy with BM25 indexing.
- **Indexable Unit Extraction**: Identifies and extracts units of documents suitable for indexing from html documents with Beautiful Soup and markdownify.
- **Utility Functions**: Includes helper functions to support the RAG system's operations.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jonathan-Mandl/RAG_System.git
   cd RAG_System

# BMS Copilot: AI-Powered Fire Protection Assistant

## Overview

An intelligent RAG-based system that answers questions from fire protection standards and building codes. Upload your PDF documents (NFPA standards, building codes, technical specs), ask questions in plain English, and get accurate answers with exact page references. It searches YOUR specific documents in real-time.

## Tools Used

- **Python** - Backend development
- **Streamlit** - Web interface
- **LangChain** - RAG pipeline orchestration
- **OpenAI GPT-3.5/4** - Answer generation
- **OpenAI Embeddings (ada-002)** - Text vectorization (1536-dim)
- **Pinecone** - Vector database for document search
- **PDFMiner.six & PDFPlumber** - PDF text extraction

## Data

Processed fire protection documentation including NFPA standards (2001, 13, etc.) and building codes. Documents are:
1. Extracted from PDFs preserving page numbers and sections
2. Chunked into 1000-character segments with 200-char overlap
3. Converted to 1536-dimensional vectors
4. Stored in Pinecone with metadata (page, section, source)

## Impact

**Fire Protection Engineers** - Find code requirements in seconds instead of hours  
**Code Officials** - Verify compliance instantly during reviews  
**Consultants** - Answer client questions in real-time    
**Facility Managers** - Access technical requirements without expertise  

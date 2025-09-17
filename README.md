# 📚 RAG PDF Chatbot (Open Source)

This project is a **Retrieval-Augmented Generation (RAG)** pipeline built with:
- **PyMuPDF (fitz)** for PDF text extraction
- **LangChain** for text splitting & QA pipeline
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) + **FAISS** for vector storage
- **Open-source LLMs** (e.g. `distilgpt2`, Falcon, Mistral, LLaMA2) for answering questions

---

## 🚀 Features
- Extracts text from PDFs
- Splits text into context chunks
- Stores chunks in FAISS for fast retrieval
- Uses HuggingFace LLM for natural language Q&A
- Fully open-source, no API keys required

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/rag-pdf-chat.git
cd rag-pdf-chat
pip install -r requirements.txt


"""
RAG PDF Chatbot using PyMuPDF + FAISS + HuggingFace
----------------------------------------------------
This script extracts text from PDFs using PyMuPDF,
splits into chunks, creates embeddings with HuggingFace,
stores them in FAISS, and answers questions via an open-source LLM.

Author: kavitha agasthya

"""

import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline


# --------------------------------------------------
# 1. Load PDF and Extract Text using PyMuPDF
# --------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Faster and lighter than pdfplumber for most cases.
    """
    doc = fitz.open(pdf_path)   # Open PDF
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]                 # Access page
        text += page.get_text("text") + "\n" # Extract text
    doc.close()
    return text


# --------------------------------------------------
# 2. Split Text into Chunks
# --------------------------------------------------
def chunk_text(text: str):
    """
    Splits long text into manageable chunks
    so embeddings and LLM can handle context better.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # max size of chunk
        chunk_overlap=200, # overlap so context is preserved
    )
    return text_splitter.split_text(text)


# --------------------------------------------------
# 3. Build FAISS Vector Database
# --------------------------------------------------
def build_vectorstore(chunks):
    """
    Converts text chunks into embeddings and stores in FAISS vector DB.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# --------------------------------------------------
# 4. Load Open Source LLM
# --------------------------------------------------
def load_llm():
    """
    Loads an open-source HuggingFace model for text generation.
    You can replace 'distilgpt2' with a stronger model (Falcon, Mistral, LLaMA2, etc).
    """
    llm_pipeline = pipeline(
        "text-generation",
        model="distilgpt2",  # small demo model; replace with larger if needed
        max_new_tokens=200,
        temperature=0.2
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)


# --------------------------------------------------
# 5. Build RetrievalQA Chain
# --------------------------------------------------
def build_qa_chain(vectorstore, llm):
    """
    Combines FAISS retriever with the LLM for Retrieval Augmented Generation.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )


# --------------------------------------------------
# 6. Run Q&A on a PDF
# --------------------------------------------------
def chat_with_pdf(pdf_path, query):
    """
    End-to-end RAG pipeline: Extract text -> Chunk -> Embed -> Retrieve -> Answer.
    """
    print(f"\n[INFO] Processing: {pdf_path}")
    
    # Step 1: Extract
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk
    chunks = chunk_text(text)
    
    # Step 3: Build Vector DB
    vectorstore = build_vectorstore(chunks)
    
    # Step 4: Load LLM
    llm = load_llm()
    
    # Step 5: Build QA chain
    qa_chain = build_qa_chain(vectorstore, llm)
    
    # Step 6: Ask question
    result = qa_chain.invoke(query)
    
    print("\nQuestion:", query)
    print("Answer:", result["result"])
    print("\nSources (chunks used):")
    for doc in result["source_documents"]:
        print("-", doc.page_content[:200], "...")


# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    # Example usage
    chat_with_pdf("sample.pdf", "What is the main conclusion of this document?")

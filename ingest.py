import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def process_documents():
    # Find all PDFs in the data folder
    pdf_files = glob.glob("./data/*.pdf")
    
    if not pdf_files:
        print("❌ Error: No PDF files found in the './data' folder! Please add your PDF there.")
        return None

    # Pick the first PDF file found in the folder
    target_pdf = pdf_files[0]
    print(f"🔄 Step 1: Loading your confidential PDF ({target_pdf})...")
    loader = PyPDFLoader(target_pdf)
    documents = loader.load()
    
    print("✂️ Step 2: Chunking document into secure text fragments...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    print(f"🧬 Step 3: Generating mathematical vectors for {len(chunks)} fragments...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("💾 Step 4: Storing data safely inside local ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("✅ Success! Your document is processed and stored locally.")
    return vector_db

if __name__ == "__main__":
    process_documents()

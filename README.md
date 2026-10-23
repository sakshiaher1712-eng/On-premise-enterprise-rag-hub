# 🔒 On-Premise Enterprise Knowledge Hub (Privacy-First RAG Platform)

A 100% local, secure **Retrieval-Augmented Generation (RAG)** platform designed for enterprise architectures. This software allows companies to upload confidential documents (Financial Audits, Legal Contracts, HR Policies) and query them through a secure web interface using offline Large Language Models (LLMs)—**ensuring zero data leakage to public clouds and zero external API costs.**

## 📈 Enterprise Value Proposition
* **Absolute Data Privacy:** Operates completely offline. Zero data leaves the physical corporate hardware, complying strictly with GDPR and data residency laws.
* **Zero Token Financial Fees:** Uses open-source local inference models via Ollama rather than charging recurring cloud API token fees.
* **Hallucination Countermeasures:** Implements custom contextual prompt engineering acting as a ground-truth constraints anchor to prevent the LLM from inventing information.

## 🏗️ System Architecture & Data Flow
1. **Document Ingestion:** Secure parsing of unstructured PDF files using `PyPDFLoader`.
2. **Deterministic Text Chunking:** Implements a `RecursiveCharacterTextSplitter` configured with a chunk size of 600 characters and a 100-character overlap to maximize contextual retention.
3. **Vector Mathematics & Embedding Engine:** Converts text fragments into geometric coordinate embeddings locally via the `all-MiniLM-L6-v2` Hugging Face pipeline.
4. **Vector Database Management:** Secure local storage indexing using `ChromaDB`.
5. **Inference Loop:** Runs semantic similarity matching queries to extract the top 3 closest document blocks, passing them into an offline `Llama 3` brain via `Ollama`.

## 🛠️ Technical Stack
* **Language:** Python
* **AI Orchestration Framework:** LangChain / LangChain-Community
* **Vector Store:** ChromaDB
* **Local Inference Hub:** Ollama (Llama 3 Model)
* **Frontend UI Layout:** Streamlit Framework

## 🚀 Installation & Local Execution
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd private-ai-hub
   ```
2. **Setup virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Boot up the Web UI application:**
   ```bash
   streamlit run app.py
   ```

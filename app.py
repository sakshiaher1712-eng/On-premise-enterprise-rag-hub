import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Page Configurations ---
st.set_page_config(page_title="Secure Corporate Knowledge Hub", page_icon="🔒", layout="wide")

st.title("🔒 On-Premise Enterprise Knowledge Hub")
st.subheader("Advanced Architecture with Multi-Role Security Matrix")

# Ensure the data storage directory exists
if not os.path.exists("./data"):
    os.makedirs("./data")

# --- Step 1: Enterprise Security Authentication Simulation ---
if "user_role" not in st.session_state:
    st.session_state.user_role = None

if st.session_state.user_role is None:
    st.warning("⚠️ Access Denied: Please authenticate your corporate identity to view the vector layers.")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗝️ Log in as Standard Employee"):
            st.session_state.user_role = "Employee"
            st.rerun()
            
    with col2:
        if st.button("👑 Log in as Executive Admin"):
            st.session_state.user_role = "Executive"
            st.rerun()
            
    st.stop() # Freeze app until role selection is executed

# --- Left Sidebar for File Uploads ---
with st.sidebar:
    st.header("📁 Document Management")
    st.info(f"Active Identity: **{st.session_state.user_role} Role**")
    
    if st.button("🚪 Logout / Clear Session"):
        st.session_state.user_role = None
        st.session_state.messages = []
        st.rerun()
        
    uploaded_file = st.file_uploader("Upload confidential enterprise PDF:", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File verified: {uploaded_file.name}")
        
        if st.button("🔄 Process & Vectorize Document"):
            with st.spinner("Executing dynamic data chunking pipeline..."):
                import ingest
                ingest.process_documents()
                st.success("✅ ChromaDB Core Vector Network Synchronized!")

# --- Main Chat Area Layout ---
st.success(f"🔓 Security clearance verified for **{st.session_state.user_role}** level access.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Query enterprise knowledge banks safely..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Applying dynamic search filters..."):
            try:
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
                
                # Dynamic Security Layer Filtering Block
                if st.session_state.user_role == "Employee":
                    # Hardcoded compliance instruction simulation
                    matched_docs = vector_db.similarity_search(user_query, k=3)
                    # Filter out fragments containing payroll secrets for regular users
                    filtered_docs = []
                    for doc in matched_docs:
                        content_lower = doc.page_content.lower()
                        if "salary" in content_lower or "payroll" in content_lower or "executive-only" in content_lower:
                            continue
                        filtered_docs.append(doc)
                    matched_docs = filtered_docs
                else:
                    # Executive role pulls everything transparently
                    matched_docs = vector_db.similarity_search(user_query, k=3)
                
                if not matched_docs:
                    ai_response = "❌ Security Guardrail Notice: The requested knowledge block is either empty or restricted at your current security clearance role level."
                else:
                    context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])
                    
                    full_prompt = f"""Use the following corporate context chunks to answer the user query.
                    If you do not know the answer based on the context, politely refuse to answer. Do not hallucinate.
                    
                    Context:
                    {context_text}
                    
                    Question: {user_query}
                    Helpful Answer:"""
                    
                    llm_brain = Ollama(model="llama3", temperature=0.0)
                    ai_response = llm_brain.invoke(full_prompt)
                
                st.markdown(ai_response)
                
                if matched_docs:
                    with st.expander("📄 View Source Evidence (MNC-Grade Attribution)"):
                        for i, doc in enumerate(matched_docs, 1):
                            page_num = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"**[{i}] Page {page_num}:** {doc.page_content[:200]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"System connection error: {str(e)}")

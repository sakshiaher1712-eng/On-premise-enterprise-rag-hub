from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

def ask_local_ai(user_question):
    print("🧠 Step 1: Loading your local database and embedding model...")
    # Load our local embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect back to your saved local database folder
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    
    print("⚡ Step 2: Running manual semantic search to fetch relevant file pages...")
    # Manually pull the top 3 closest matching text blocks from your PDF
    matched_docs = vector_db.similarity_search(user_question, k=3)
    
    # Combine the matching pieces of text into one large context block
    context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])
    
    print("🔒 Step 3: Formatting strict privacy prompt...")
    # Create the custom instruction template manually
    full_prompt = f"""Use the following pieces of corporate context to answer the question at the end. 
    If you do not know the answer, say that you do not know. Do not try to make up or hallucinate an answer.
    Keep the answer concise and professional.

    Context:
    {context_text}

    Question: {user_question}
    Helpful Answer:"""

    print("🤖 Step 4: Invoking local Ollama (Llama 3 Brain)...")
    # Initialize your offline Llama 3 brain
    llm_brain = Ollama(model="llama3", temperature=0.0)
    
    # Pass the prompt directly to the AI
    ai_response = llm_brain.invoke(full_prompt)
    
    print("\n=================== AI RESPONSE ===================")
    print(ai_response)
    print("===================================================\n")
    
    print("📄 EXACT PAGES USED AS SOURCE GROUND TRUTH:")
    for i, doc in enumerate(matched_docs, 1):
        page_num = doc.metadata.get('page', 'Unknown')
        print(f"[{i}] From Page {page_num}: {doc.page_content[:120]}...")
        
    return ai_response

if __name__ == "__main__":
    print("Testing the clean query infrastructure...")
    test_query = "What is the summary of this candidate's skills and experience?"
    ask_local_ai(test_query)

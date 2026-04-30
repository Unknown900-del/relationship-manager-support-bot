import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(
    temperature=0, 
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile" 
)

def get_answer(question, use_history=False):
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_collection(name="banking_bot")
    
    # --- STEP 1: CONTEXTUALIZE QUESTION (If Memory is ON) ---
    search_query = question
    if use_history and len(st.session_state.messages) > 0:
        # Get the last 3 turns to keep context sharp
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
        
        condense_prompt = f"""
        Given the following chat history and a follow-up question, rephrase it as a 
        standalone search query for a banking product brochure. 
        Output ONLY the search query. No preamble.

        CHAT HISTORY:
        {history_str}

        FOLLOW-UP: {question}"""
        
        condensed_res = llm.invoke(condense_prompt)
        search_query = condensed_res.content
        print(f"DEBUG: Standalone Query: {search_query}")

    # --- STEP 2: BANK DETECTION ---
    q_lower = search_query.lower()
    target_bank = None
    if "sbi" in q_lower: target_bank = "SBI"
    elif "hdfc" in q_lower: target_bank = "HDFC"
    elif "jpmorgan" in q_lower: target_bank = "JPMorgan"

    query_vector = embed_model.encode(search_query).tolist()

    # --- STEP 3: DEEP RETRIEVAL ---
    # Increased n_results to 15 to ensure brochures aren't hidden by Annual Reports
    try:
        if target_bank:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=15, 
                where={"bank": {"$in": banks_mentioned}}
            )
            # Fallback if the metadata filter is too strict
            if not results['documents'] or len(results['documents'][0]) == 0:
                results = collection.query(query_embeddings=[query_vector], n_results=15)
        else:
            results = collection.query(query_embeddings=[query_vector], n_results=15)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        results = collection.query(query_embeddings=[query_vector], n_results=15)

    # --- STEP 4: BUILD CONTEXT ---
    context_parts = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            source = results['metadatas'][0][i].get('source', 'Unknown')
            bank = results['metadatas'][0][i].get('bank', 'Unknown')
            context_parts.append(f"--- SOURCE: {bank} ({source}) ---\n{text}")
    
    context = "\n\n".join(context_parts)

    # --- STEP 5: FINAL RM PROMPT ---
    prompt = f"""
    You are a Senior Relationship Manager Support Bot. Provide accurate data based ONLY on the context.
    
    ### RULES:
    1. Answer ONLY for the specific bank requested.
    2. If details are missing, state: "Product details not in current database."
    3. **BOLD** all numbers and product names.
    4. State the filename source at the top.

    ### CONTEXT:
    {context}

    ### CLIENT QUESTION:
    {question}

    ### EXPERT RM RECOMMENDATION:"""
    
    return llm.invoke(prompt)
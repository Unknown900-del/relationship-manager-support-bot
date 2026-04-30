import os
from dotenv import load_dotenv
load_dotenv()
# Ensure this path is correct for your system
os.environ['STREAMLIT_CONFIG_DIR'] = r'F:\.streamlit'
import streamlit as st
from brain import get_answer

st.set_page_config(page_title="RM Support", layout="centered")
st.title("🏦 Banking RM Support Bot")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("System Settings")
    # This is your key feature for the SME demo
    enable_memory = st.checkbox("Enable Contextual Memory", value=False)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input logic
if prompt := st.chat_input("Ask about SBI, HDFC, or product details..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Analyzing reports and brochures..."):
        # We pass the memory toggle state to the brain
        response = get_answer(prompt, use_history=enable_memory)
        
        # Extract content from the Groq message object
        display_text = response.content
        
        st.session_state.messages.append({"role": "assistant", "content": display_text})
        st.chat_message("assistant").write(display_text)
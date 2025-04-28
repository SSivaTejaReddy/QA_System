import streamlit as st
from rag_pipeline_groq import QASystem
from language_config import supported_languages

# Configure page
st.set_page_config(page_title=" Q&A", page_icon="🏥")
st.title("Chatbot Q&A Sysytem")

#@st.cache_resource
def load_qa():
    return QASystem() or st.error("⚠️ System initialization failed")

# Initialize chat with welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help?"}]

# Display messages (using default avatars)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Add a selectbox to pick task before sending a message
task = st.selectbox("Choose a task for this input:", options=["qa", "summary", "translation"])

if task == "translation":
    source_language = st.selectbox("Selct source language:",options=supported_languages)
    target_language = st.selectbox("Selct target language:",options=supported_languages)
else:
    source_language = None
    target_language = None

# Process user input
if prompt := st.chat_input("💬 Post a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"), st.spinner("🔍 Searching..."):
        try:
            response = load_qa().query(prompt,task,source_language, target_language)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error = f"Error: {str(e)}"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})

if st.button("Clear History"):
    st.session_state.messages = []
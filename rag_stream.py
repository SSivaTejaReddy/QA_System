import streamlit as st
from rag_pipeline_groq import QASystem

# Configure page
st.set_page_config(page_title=" Q&A", page_icon="🏥")
st.title("Chatbot Q&A Sysytem")

@st.cache_resource
def load_qa():
    return QASystem() or st.error("⚠️ System initialization failed")

# Initialize chat with welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help?"}]

# Display messages (using default avatars)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process user input
if prompt := st.chat_input("💬 Post a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"), st.spinner("🔍 Searching..."):
        try:
            response = load_qa().query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error = f"Error: {str(e)}"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})

if st.button("Clear History"):
    st.session_state.messages = []
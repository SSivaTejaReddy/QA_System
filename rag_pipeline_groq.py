import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Groq client
client = Groq(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")  # Replace with your key

class QASystem:
    def __init__(self, index_path="faiss_index"):
        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Load FAISS vectorstore
        self.vectorstore = FAISS.load_local(
            folder_path=index_path,
            embeddings=self.embeddings,
            #allow_dangerous_deserialization=True
        )

        # Prompt template for RAG
        self.prompt_template = """
You are a helpful assistant. Use only the context below to answer the question accurately and completely.
Include detailed information about answer and relevant information related to it.
If you cannot answer based on the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    
    def query(self, question):
        # Retrieve relevant context from FAISS
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

        # Format the Groq-compatible prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.prompt_template.format(context=context, question=question)}
        ]

        # Call Groq API
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # "mixtral-8x7b-32768" "llama3-70b-8192" "llama-3.1-8b-instant"
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    qa = QASystem()
    print("\nQ&A Chat (type 'exit' or 'quit' to stop)\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_question:
            continue

        answer = qa.query(user_question)
        print("\nAnswer:\n" + answer + "\n" + "-"*50 + "\n")
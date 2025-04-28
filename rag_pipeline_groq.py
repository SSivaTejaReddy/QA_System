import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from prompt import Question_Answering, Summaraziation, Translation
from language_config import supported_languages
from choose_language import choose_language

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Groq client
client = Groq(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")  # Replace with your key

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
            allow_dangerous_deserialization=True
        )

        # Prompt template
        self.prompts= {
            "qa": Question_Answering,
            "summary" : Summaraziation,
            "translation" : Translation
        }


    def query(self, question, ptask, source_language, target_language):

        prompt_input = self.prompts.get(ptask, Question_Answering)
        
        if ptask == "summary":
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = prompt_input.format(content=context)
        elif ptask == "translation":
            prompt = prompt_input.format(content=question, source_language = source_language, target_language = target_language)
        else:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = prompt_input.format(context=context, question=question)

        # Format the Groq-compatible prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
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
        ptask = input("Choose task (qa/summary/translation): ").strip().lower()
        if ptask in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        user_question = input("Your question: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_question:
            continue

        target_language = None
        source_language = None
        if ptask == "translation":
            source_language = choose_language(supported_languages)
            target_language = choose_language(supported_languages)


        answer = qa.query(user_question, ptask, source_language, target_language)
        print("\nAnswer:\n" + answer + "\n" + "-"*50 + "\n")
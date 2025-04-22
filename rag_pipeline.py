from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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

        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model="llama3",
            temperature=0.7,
            num_ctx=4096
        )

        # Refined prompt template for a more complete answer
        self.prompt_template = PromptTemplate.from_template(
            """
You are a helpful assistant. Use only the context below to answer the question accurately and completely.
Include detailed information about the person's background, family health history, and any relevant actions they took.
If you cannot answer based on the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # Create RetrievalQA chain with the prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",  # You can change this to "mmr" if needed for diversity
                search_kwargs={"k": 4}  # Increase 'k' to retrieve more context
            ),
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def query(self, question):
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

if __name__ == "__main__":
    qa = QASystem()
    print("\n Q&A Chat (type 'exit' or 'quit' to stop)\n")

    while True:
        user_question = input("Your question: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_question:
            continue

        answer = qa.query(user_question)
        print("\nAnswer:\n" + answer + "\n" + "-"*50 + "\n")

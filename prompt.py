from langchain.prompts import PromptTemplate

Question_Answering = PromptTemplate.from_template("""
You are a helpful assistant. Use only the context below to answer the question accurately and completely.
Include detailed information about answer and relevant information related to it.
If you cannot answer based on the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

Summaraziation = PromptTemplate.from_template("""
Summarize the following content concisely and clearly.

Content:
{content}

Summary:
""")

Translation = PromptTemplate.from_template("""
Translate the following content {source_language} to {target_language}.

Question:
{content}

Translation:
""")
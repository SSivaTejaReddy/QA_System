from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def create_embeddings(chunks, db_path="faiss_index"):
    """
    Create and save FAISS vector store from document chunks.

    Args:
        chunks (List[Document]): List of document chunks
        db_path (str): Path to save the FAISS index

    Returns:
        FAISS: Vector store instance
    """
    try:
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("Creating vector store...")
        
        # Manual progress bar with embeddings
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        
        embedded_chunks = []
        for text in tqdm(texts, desc="Embedding documents"):
            embedded_chunks.append(text)

        vectorstore = FAISS.from_texts(texts=embedded_chunks, embedding=embeddings, metadatas=metadatas)

        os.makedirs(db_path, exist_ok=True)
        vectorstore.save_local(db_path)
        logger.info(f"Vector store saved to {db_path}")
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from load_docs import load_and_chunk_pdfs

    chunks = load_and_chunk_pdfs()
    if chunks:
        create_embeddings(chunks)

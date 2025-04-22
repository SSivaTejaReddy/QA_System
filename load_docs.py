import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def load_and_chunk_pdfs(folder_path="docs"):
    """
    Load and chunk PDF documents from specified folder.
    
    Args:
        folder_path (str): Path to folder containing PDFs
        
    Returns:
        List[Document]: List of chunked documents
    """
    all_docs = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(folder_path, file)
                logger.info(f"Loading: {file_path}")
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                logger.info(f"Pages loaded: {len(pages)}")
                all_docs.extend(pages)

        logger.info(f"Total documents loaded: {len(all_docs)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=70,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in document loading: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder_path = "docs"
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
    else:
        load_and_chunk_pdfs(folder_path)
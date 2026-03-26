from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf_chunks(path, chunk_size=800, chunk_overlap=150):
    """
    Load PDF and split into chunks with preserved metadata.
    
    Args:
        path (str): Path to PDF file
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of Document objects with metadata
    """
    try:
        logger.info(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")
        
        return chunks
        
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return []
    
    except Exception as e:
        logger.error(f"Error loading PDF {path}: {str(e)}")
        return []
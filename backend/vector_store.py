import os
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*sparse_softmax.*')
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

import chromadb
from chromadb.utils import embedding_functions
from utils.document_loader import load_pdf_chunks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
client = chromadb.PersistentClient(path=db_path)

nec_collection = client.get_or_create_collection(
    name="skin_cancer_kb",
    embedding_function=embedding,
)


def build_vector_db():
    """Build and populate the vector database with document embeddings."""
    
    logger.info("Loading and indexing documents...")
    logger.info(f"Database path: {db_path}")
    
    # Load skin cancer knowledge base
    kb_docs = load_pdf_chunks("data/skin_cancer_chatbot_QA_1000.pdf")

    logger.info(f"Indexing {len(kb_docs)} skin cancer knowledge base chunks...")
    for i, doc in enumerate(kb_docs):
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata['source_type'] = 'Skin Cancer Knowledge Base'
        nec_collection.add(
            documents=[doc.page_content],
            ids=[f"skin_cancer_{i}"],
            metadatas=[metadata]
        )
    
    logger.info("Vector database indexing complete!")
    logger.info(f"✅ Skin Cancer KB collection has {nec_collection.count()} documents")
    logger.info(f"📁 Embeddings persisted to: {db_path}")
    logger.info(f"📁 Embeddings persisted to: {db_path}")


def search_nec(query):
    """Search skin cancer knowledge base."""
    results = nec_collection.query(
        query_texts=[query],
        n_results=5
    )
    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else []
    }


def search_solar(query):
    """Alias for search_nec - returns empty for backward compatibility."""
    return {
        "documents": [],
        "metadatas": []
    }


def search_wattmonk(query):
    """Alias for search_nec - returns empty for backward compatibility."""
    return {
        "documents": [],
        "metadatas": []
    }
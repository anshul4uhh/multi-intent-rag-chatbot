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

nec_collection = client.create_collection(
    name="nec_docs",
    embedding_function=embedding,
    get_or_create=True
)

solar_collection = client.create_collection(
    name="solar_docs",
    embedding_function=embedding,
    get_or_create=True
)

wattmonk_collection = client.create_collection(
    name="wattmonk_docs",
    embedding_function=embedding,
    get_or_create=True
)


def build_vector_db():
    """Build and populate the vector database with document embeddings."""
    
    logger.info("Loading and indexing documents...")
    logger.info(f"Database path: {db_path}")
    
    nec_docs = load_pdf_chunks("data/nec.pdf")
    solar_docs = load_pdf_chunks("data/solar-power-installation.pdf")
    wattmonk_docs = load_pdf_chunks("data/wattmonk.pdf")

    
    logger.info(f"Indexing {len(nec_docs)} NEC document chunks...")
    for i, doc in enumerate(nec_docs):
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata['source_type'] = 'NEC Electrical Code'
        nec_collection.add(
            documents=[doc.page_content],
            ids=[f"nec_{i}"],
            metadatas=[metadata]
        )

    
    logger.info(f"Indexing {len(solar_docs)} Solar document chunks...")
    for i, doc in enumerate(solar_docs):
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata['source_type'] = 'Solar Installation Manual'
        solar_collection.add(
            documents=[doc.page_content],
            ids=[f"solar_{i}"],
            metadatas=[metadata]
        )

    # Index Wattmonk documents
    logger.info(f"Indexing {len(wattmonk_docs)} Wattmonk document chunks...")
    for i, doc in enumerate(wattmonk_docs):
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata['source_type'] = 'Wattmonk Company Information'
        wattmonk_collection.add(
            documents=[doc.page_content],
            ids=[f"wattmonk_{i}"],
            metadatas=[metadata]
        )
    
    logger.info("Vector database indexing complete!")
    
    
    logger.info(f"✅ NEC collection has {nec_collection.count()} documents")
    logger.info(f"✅ Solar collection has {solar_collection.count()} documents")
    logger.info(f"✅ Wattmonk collection has {wattmonk_collection.count()} documents")
    logger.info(f"📁 Embeddings persisted to: {db_path}")


def search_nec(query):
    """Search NEC Electrical Code documents."""
    results = nec_collection.query(
        query_texts=[query],
        n_results=3
    )
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0] if results["metadatas"] else []
    }


def search_solar(query):
    """Search Solar Installation documents."""
    results = solar_collection.query(
        query_texts=[query],
        n_results=3
    )
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0] if results["metadatas"] else []
    }


def search_wattmonk(query):
    """Search Wattmonk Company documents."""
    results = wattmonk_collection.query(
        query_texts=[query],
        n_results=3
    )
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0] if results["metadatas"] else []
    }
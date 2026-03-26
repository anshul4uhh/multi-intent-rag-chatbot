import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF startup logs
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*sparse_softmax.*')
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

from backend.vector_store import build_vector_db
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Starting Vector Database Build Process")
    logger.info("=" * 60)
    
    try:
        logger.info("\n📚 Building vector database with semantic embeddings...")
        logger.info("This may take a few minutes on first run.\n")
        
        build_vector_db()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Vector Database Successfully Built!")
        logger.info("=" * 60)
        logger.info("\nDatabase ready for chatbot queries.")
        logger.info("Run 'streamlit run app.py' to start the chatbot.\n")
        
    except Exception as e:
        logger.error(f"❌ Error building vector database: {str(e)}")
        logger.error("Please check that all PDF files exist in the data/ directory")
        raise

if __name__ == "__main__":
    main()
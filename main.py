# main.py
from config import Config
from document_processor import DocumentProcessor
from neo4j_manager import Neo4jManager
from embeddings_manager import EmbeddingsManager
from rag_pipeline import RAGPipeline
import logging
import time

def main():
    try:
        # Initialize components
        config = Config()
        doc_processor = DocumentProcessor(config)
        neo4j_manager = Neo4jManager(config)
        embeddings_manager = EmbeddingsManager(config)
        rag = RAGPipeline(config)

        # Load and process document
        documents = doc_processor.load_document("devopssetup.txt")
        logging.info(f"Loaded {len(documents)} document chunks")

        # Process documents to extract entities and relationships
        graph_documents = rag.process_documents(documents)
        logging.info(f"Processed {len(graph_documents)} documents for graph storage")

        # Add to Neo4j and create embeddings
        neo4j_manager.add_graph_documents(graph_documents)
        vector_retriever = embeddings_manager.create_vector_index()

        # Setup retrieval chain
        rag.setup_retrieval_chain(vector_retriever)

        # Test the system
        question = "What are the main components mentioned in the setup?"
        answer = rag.chain.invoke(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    except Exception as e:
        logging.error(f"Failed to initialize RAG pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
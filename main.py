# main.py
from config import Config
from rag_facade import RAGFacade
import logging
import time

def main():
    try:
        # Initialize RAG Facade
        rag_facade = RAGFacade()
        rag = rag_facade.process_documents("devopssetup.txt")

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

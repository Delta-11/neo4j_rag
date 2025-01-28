import os
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session, Transaction
from neo4j.exceptions import ClientError, TransactionError
from custom_embedder import OllamaEmbedder, OllamaNeo4jRM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.embedder = OllamaEmbedder(model="granite-embedding:278m")
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
            
    def check_vector_index(self, session: Session) -> bool:
        """Check if vector index exists."""
        try:
            result = session.run("""
                SHOW INDEXES
                WHERE type = 'VECTOR' AND name = 'vector'
            """)
            return any(result)
        except Exception as e:
            logger.error(f"Error checking vector index: {str(e)}")
            raise
            
    def create_vector_index(self, session: Session) -> None:
        """Create vector index if it doesn't exist."""
        try:
            if not self.check_vector_index(session):
                session.run("""
                    CALL db.index.vector.createNodeIndex(
                        'vector',
                        'Document',
                        'embedding',
                        768,
                        'cosine'
                    )
                """)
                logger.info("Vector index created successfully")
            else:
                logger.info("Vector index already exists")
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            raise
            
    def add_document(self, tx: Transaction, text: str, embedding: np.ndarray) -> None:
        """Add a document with its embedding to Neo4j."""
        try:
            tx.run("""
                MERGE (d:Document {text: $text})
                SET d.embedding = $embedding
            """, text=text, embedding=embedding.tolist())
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
            
    def add_sample_documents(self, session: Session) -> None:
        """Add sample documents with embeddings."""
        documents = [
            "Quantum computing leverages quantum mechanics to perform complex calculations.",
            "The significance of quantum computing lies in its potential to solve problems that classical computers cannot.",
            "Quantum entanglement and superposition are key principles in quantum computing."
        ]
        
        try:
            # Get embeddings for all documents
            embeddings = self.embedder(documents)
            
            # Add documents in a single transaction
            def create_documents(tx):
                for doc, emb in zip(documents, embeddings):
                    self.add_document(tx, doc, emb)
                    
            session.execute_write(create_documents)
            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add sample documents: {str(e)}")
            raise
            
    def setup_database(self) -> None:
        """Set up the Neo4j database with required index and sample data."""
        try:
            self.connect()
            with self.driver.session() as session:
                self.create_vector_index(session)
                self.add_sample_documents(session)
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise
        finally:
            self.close()
            
def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Neo4j manager
    neo4j_manager = Neo4jManager(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    
    try:
        # Setup database
        neo4j_manager.setup_database()
        
        # Initialize retriever
        retriever = OllamaNeo4jRM(
            index_name="vector",
            text_node_property="text",
            embedding_model="granite-embedding:278m",
            k=3
        )
        
        # Test retrieval
        query = "Explore the significance of quantum computing"
        logger.info(f"Testing retrieval with query: {query}")
        results = retriever(query, k=3)
        
        if results:
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                if isinstance(result, (list, tuple)):
                    text, score = result
                    print(f"\nResult {i}:")
                    print(f"Text: {text}")
                    print(f"Similarity Score: {score:.4f}")
                else:
                    print(f"\nResult {i}: {result}")
        else:
            print("\nNo matching documents found")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

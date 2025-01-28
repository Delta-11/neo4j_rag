# config.py
import os
from dotenv import load_dotenv

class Config:
    """Configuration management for the RAG system."""
    
    def __init__(self):
        """
        Initialize the configuration object.

        This method will load configuration values from environment variables
        and set default values if they are not provided.

        Configuration values are:

        - NEO4J_URI: The URI to connect to the Neo4j instance. Default is
          "bolt://localhost:7687".
        - NEO4J_USER: The username to use when connecting to Neo4j. Default is
          "neo4j".
        - NEO4J_PASSWORD: The password to use when connecting to Neo4j. Default
          is "password".
        - LLM_MODEL: The model to use for the LLM. Default is "deepseek-r1:1.5b".
        - EMBEDDING_MODEL: The model to use for the embeddings. Default is
          "granite-embedding:278m".
        - CHUNK_SIZE: The size of the chunks to use for document processing.
          Default is 200.
        - CHUNK_OVERLAP: The amount of overlap between chunks for document
          processing. Default is 50.
        """
        load_dotenv()
        
        # Neo4j Configuration
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        
        # LLM Configuration
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "granite-embedding:278m")
        
        # Document Processing Configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
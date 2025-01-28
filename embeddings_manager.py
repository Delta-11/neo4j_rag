# embeddings_manager.py
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

class EmbeddingsManager:
    """Manages vector embeddings and retrieval."""
    
    def __init__(self, config):
        """
        Initialize the EmbeddingsManager.

        Args:
            config: Configuration object containing settings for embedding and vector index management.

        This method initializes the EmbeddingsManager by setting the configuration, 
        and initializing the embeddings and vector index attributes to None. 
        It also sets up the embeddings by calling the setup_embeddings method.
        """

        self.config = config
        self.embeddings = None
        self.vector_index = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        
        """
        Set up the embeddings by initializing the OllamaEmbeddings model with the
        embedding model specified in the configuration.

        If the initialization fails, a logging error is raised with the error message.
        """
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.config.EMBEDDING_MODEL
            )
        except Exception as e:
            logging.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def create_vector_index(self):
        
        """
        Create a vector index using the existing graph in Neo4j.

        This method initializes a Neo4jVector from the existing graph by using
        the provided embeddings. It configures the vector index to use a hybrid
        search type, focusing on nodes labeled 'Document' and properties like
        'text' for text nodes and 'embedding' for embedding nodes. The method
        returns the vector index as a retriever for further use.

        Raises:
            Exception: If the vector index creation fails, an error message is logged
            and the exception is re-raised.
        """

        try:
            self.vector_index = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding",
            )
            return self.vector_index.as_retriever()
        except Exception as e:
            logging.error(f"Failed to create vector index: {str(e)}")
            raise
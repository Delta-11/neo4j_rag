# neo4j_manager.py
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
from typing import List
from langchain_core.documents import Document
import logging

class Neo4jManager:
    """Manages Neo4j database operations."""
    
    def __init__(self, config):
        """
        Initialize the Neo4jManager with the given configuration.

        This constructor sets up the configuration, initializes the driver and graph
        attributes to None, and establishes a connection to the Neo4j database by 
        invoking the setup_connection method.

        Args:
            config: A configuration object containing Neo4j connection details.
        """

        self.config = config
        self.driver = None
        self.graph = None
        self.setup_connection()
    
    def setup_connection(self):
        """
        Establishes a connection to the Neo4j database.

        This method creates a connection to the Neo4j database using the
        configuration provided, and initializes the graph attribute with a
        Neo4jGraph object. It also creates necessary indices in the database.

        If the connection or index creation fails, a logging error is raised
        with the error message.
        """
        try:
            self.driver = GraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            self.graph = Neo4jGraph()
            self.create_indices()
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def create_indices(self):
        """Create necessary indices in Neo4j."""
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_fulltext_index)
                logging.info("Index created or already exists")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                logging.info("Index already exists")
            else:
                logging.error(f"Error creating index: {str(e)}")
    
    def _create_fulltext_index(self, tx):
        
        """
        Create a fulltext index on the 'id' property of nodes labeled '__Entity__'.

        This method constructs and executes a Cypher query to create a fulltext index
        named 'fulltext_entity_id' for the 'id' property on all nodes with the label
        '__Entity__'. The transaction is executed within the given Neo4j transaction context.

        Args:
            tx: A Neo4j transaction object used to execute the query.
        """

        query = '''
        CREATE FULLTEXT INDEX fulltext_entity_id 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)
    
    def add_graph_documents(self, graph_documents: List[Document]):
        """
        Adds the given list of Document objects to the Neo4j graph.

        Args:
            graph_documents (List[Document]): A list of Document objects to be added to the graph.

        Raises:
            Exception: If the addition of documents to the graph fails.
        """
        try:
            # Create Cypher queries to add documents and their relationships
            with self.driver.session() as session:
                for doc in graph_documents:
                    # Create document node
                    create_doc_query = """
                    CREATE (d:Document {
                        content: $content,
                        source: $source,
                        entity_type: $entity_type,
                        entity_value: $entity_value,
                        entity_text: $entity_text
                    })
                    """
                    
                    session.run(
                        create_doc_query,
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'unknown'),
                        entity_type=doc.metadata.get('entity_type', ''),
                        entity_value=doc.metadata.get('entity_value', ''),
                        entity_text=doc.metadata.get('entity_text', '')
                    )
                    
        except Exception as e:
            logging.error(f"Failed to add documents to graph: {str(e)}")
            raise
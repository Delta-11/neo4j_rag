# Imports and Environment Setup: The script imports necessary libraries and loads environment variables.
# Language Model Initialization: It initializes an Azure OpenAI language model using the loaded environment variables.
# Text Loading and Splitting: The script loads a text file, splits it into chunks, and processes these chunks into documents.
# Graph Transformation: It uses the language model to transform the documents into graph structures (nodes and relationships).
# Neo4j Graph Initialization: The script initializes a connection to a Neo4j graph database.
# Allowed Relationships and Nodes: It defines allowed relationships and nodes for the graph.
# Graph Document Filtering: The script filters the graph documents based on allowed nodes and relationships.
# Graph Insertion: Finally, it inserts the filtered graph documents into the Neo4j database.

# ***************************Testing of the llm model***************************
# The following code is commented out and was used for testing the language model
# system_prompt = (
#         "system",
#         "You are a helpful assistant that helps people find information.",
#     )
# user_query = ("human", "What is the capital of France?")
# messages = [system_prompt, user_query]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)
# ***************************Testing of the llm model***************************


# Import necessary libraries
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import logging
from config import AzureOpenAIModel, Neo4jGraphDB
import dspy


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_chunk_file(file_path, chunk_size=200, chunk_overlap=50):
    """
    Load the text data from a file and split it into chunks.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of Document objects.
    """
    try:
        logger.info(f"Loading text data from file: {file_path}")
        loader = TextLoader(file_path)
        document = loader.load()

        logger.info("Splitting the loaded text into chunks")
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(document)
        logger.info(f"Created {len(chunks)} chunks from the loaded text")

        logger.info("Wrapping each chunk of text into a Document object")
        documents = [Document(page_content=chunk.page_content) for chunk in chunks]
        return documents
    except Exception as e:
        logger.error(f"Error loading and chunking file: {e}")
        raise

def convert_to_graph(documents):
    """
    Convert the documents into graph documents (nodes and relationships).

    Args:
        documents (list): List of Document objects.

    Returns:
        tuple: Nodes and relationships extracted from the graph documents.
    """
    try:
        llm = AzureOpenAIModel().get_model()
        llm_transformer = LLMGraphTransformer(llm=llm)

        logger.info("Converting documents into graph documents")
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        nodes = []
        relationships = []

        unique_nodes = set()
        unique_relationships = set()

        for graph_document in graph_documents:
            for node in graph_document.nodes:
                node_type_lower = node.type.lower()
                if node_type_lower not in unique_nodes:
                    unique_nodes.add(node_type_lower)
                    nodes.append(node)

            for relationship in graph_document.relationships:
                relationship_type_lower = relationship.type.lower()
                if relationship_type_lower not in unique_relationships:
                    unique_relationships.add(relationship_type_lower)
                    relationships.append(relationship)

        logger.info(f"Extracted {len(nodes)} unique nodes and {len(relationships)} unique relationships")
        return nodes, relationships
    except Exception as e:
        logger.error(f"Error converting documents to graph documents: {e}")
        raise



def initialize_graph_transformer_and_push_to_neo4j(
    documents, allowed_nodes, allowed_relationships
):
    """
    Initialize the graph transformer with allowed nodes and relationships,
    convert the documents into filtered graph documents, and push them to Neo4j.

    Args:
        documents (list): List of Document objects.
        allowed_nodes (list): List of allowed node types.
        allowed_relationships (list): List of allowed relationships.
    """
    try:
        llm = AzureOpenAIModel().get_model()
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
        )

        logger.info("Converting documents into filtered graph documents")
        graph_documents_filtered = llm_transformer.convert_to_graph_documents(documents)

        graph = Neo4jGraphDB().get_graph()

        logger.info("Adding filtered graph documents to Neo4j database")
        graph.add_graph_documents(
            graph_documents_filtered, baseEntityLabel=True, include_source=True
        )
    except Exception as e:
        logger.error(f"Error initializing graph transformer and pushing to Neo4j: {e}")
        raise


def generate_allowed_relationships(relationships):
    """
    Generate allowed relationships using the AzureOpenAI model.

    Args:
        relationships (list): List of relationship types.

    Returns:
        list: List of allowed relationships in the format (source, relationship, target).
    """
    try:
        llm = AzureOpenAIModel().get_model()
        prompt = f"Generate allowed relationships from the following list of relationships: {relationships}"
        response = llm.invoke(prompt)
        allowed_relationships = dspy.parse(response.content)
        logger.info(f"Generated allowed relationships: {allowed_relationships}")
        return allowed_relationships
    except Exception as e:
        logger.error(f"Error generating allowed relationships: {e}")
        raise

def pipeline_to_push_documents_to_graph():
    try:
        # Load and chunk the file
        documents = load_and_chunk_file("section.txt")

        # Convert documents to graph documents
        nodes, relationships = convert_to_graph(documents)

        # Define allowed relationships for the graph
        allowed_relationships = [
            ("Service", "STORES", "Image"),
            ("Service", "MANAGES", "Registry"),
            ("Service", "PUSHED_TO", "App"),
            ("App", "USES", "Storage"),
            ("Registry", "ACCESS", "File"),
            ("File", "BELONGS_TO", "Resource group"),
        ]
        
        # Initialize the graph transformer and push data to Neo4j
        initialize_graph_transformer_and_push_to_neo4j(
            documents, nodes, allowed_relationships
        )

        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}")




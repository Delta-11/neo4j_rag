import logging
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class EnvironmentConfig(metaclass=SingletonMeta):
    def __init__(self):
        load_dotenv()
        self.config = {
            "api_version": os.getenv("API_VERSION"),
            "model_name": os.getenv("MODEL_NAME"),
            "api_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "neo4j_uri": os.getenv("NEO4J_URI"),
            "neo4j_username": os.getenv("NEO4J_USERNAME"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        }
        logger.info("Environment variables loaded and cached.")

    def get(self, key):
        return self.config.get(key)


class AzureOpenAIModel(metaclass=SingletonMeta):
    def __init__(self):
        config = EnvironmentConfig()
        self.llm = AzureChatOpenAI(
            azure_deployment=config.get("model_name"),
            api_version=config.get("api_version"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        logger.info("Azure OpenAI language model initialized.")

    def get_model(self):
        return self.llm


class Neo4jGraphDB(metaclass=SingletonMeta):
    def __init__(self):
        config = EnvironmentConfig()
        self.graph = Neo4jGraph(
            url=config.get("neo4j_uri"),
            username=config.get("neo4j_username"),
            password=config.get("neo4j_password"),
        )
        logger.info("Neo4j graph database connection initialized.")

    def get_graph(self):
        return self.graph

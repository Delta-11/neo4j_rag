# rag_pipeline.py

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for RAG Pipeline."""
    LLM_MODEL: str
    NEO4J_URI: str
    NEO4J_USER: str 
    NEO4J_PASSWORD: str

class Entity(BaseModel):
    """Entity model for extracted information."""
    text: str
    type: Optional[str] = None
    value: str  # Change this to enforce string values only

class Entities(BaseModel):
    """Container for extracted entities."""
    entities: List[Entity] = Field(default_factory=list)

class RAGPipeline:
    """RAG Pipeline for document processing and question answering."""

    def __init__(self, config: Config):
        """Initialize RAG Pipeline with configuration.
        
        Args:
            config: Configuration object containing necessary parameters
        """
        self.config = config
        self.llm = None
        self.entity_chain = None
        self.retriever = None
        self.chain = None
        self.graph = None
        
        # Initialize components
        self._setup_graph()
        self._setup_llm()
        self._setup_entity_chain()

    def _setup_graph(self) -> None:
        """Initialize Neo4j graph connection."""
        try:
            self.graph = Neo4jGraph(
                url=self.config.NEO4J_URI,
                username=self.config.NEO4J_USER,
                password=self.config.NEO4J_PASSWORD
            )
        except Exception as e:
            logger.error(f"Failed to setup Neo4j graph: {str(e)}")
            raise

    def _setup_llm(self) -> None:
        """Initialize the language model."""
        try:
            self.llm = ChatOllama(
                model=self.config.LLM_MODEL,
                temperature=0
            )
        except Exception as e:
            logger.error(f"Failed to setup LLM: {str(e)}")
            raise

    def _setup_entity_chain(self) -> None:
        """Initialize the entity extraction chain."""
        try:
            entity_prompt = """Extract important information from the following text. Look for:
            - Credentials (usernames, passwords)
            - URLs and endpoints
            - API tokens or keys
            - Server configurations
            - Environment variables
            - IP addresses
            - File paths
            - Commands

            For each piece of information, provide its value and type.

            Text: {text}

            Return the response in the following JSON format:
            {{
                "entities": [
                    {{
                        "text": "extracted text",
                        "type": "credential/url/token/config/etc",
                        "value": "the actual value"
                    }}
                ]
            }}"""
            
            parser = PydanticOutputParser(pydantic_object=Entities)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert at extracting structured information from text."),
                ("user", entity_prompt)
            ])
            
            self.entity_chain = prompt | self.llm | parser
        
        except Exception as e:
            logger.error(f"Failed to setup entity chain: {str(e)}")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        
        for doc in documents:
            try:
                # Extract entities
                result = self.entity_chain.invoke({"text": doc.page_content})
                
                # Process each entity
                for entity in result.entities:
                    # Create metadata
                    metadata = {
                        "entity_type": entity.type,
                        "entity_value": entity.value,
                        "entity_text": entity.text,
                        "source": doc.metadata.get("source", "unknown")
                    }
                    
                    # Create a new Document object
                    processed_doc = Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                    processed_docs.append(processed_doc)
                    
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
        
        return processed_docs

    def setup_retrieval_chain(self, retriever: Any) -> None:
        """Setup the retrieval chain for answering questions.
        
        Args:
            retriever: Vector store retriever for document lookup
        """
        self.retriever = retriever
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.chain = (
            {
                "context": self.full_retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def full_retriever(self, question: str) -> str:
        """Combine graph and vector retrieval results.
        
        Args:
            question: User question to process
            
        Returns:
            Combined context from graph and vector stores
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call setup_retrieval_chain first.")
            
        try:
            # Get entities from the question
            result = self.entity_chain.invoke({"text": question})
            entity_names = [entity.text for entity in result.entities]
            
            # Get graph results
            graph_results = self._get_graph_context(entity_names)
            
            # Get vector results
            vector_results = [
                el.page_content 
                for el in self.retriever.invoke(question)
            ]
            
            # Combine results
            return f"""Graph context:
                {graph_results}
                
                Vector context:
                {"#Document ".join(vector_results)}"""
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise

    def _get_graph_context(self, entities: List[str]) -> str:
        """Get context from the graph database.
        
        Args:
            entities: List of entity names to query
            
        Returns:
            String containing relationship information between entities
        """
        if not entities:
            return "No relevant entities found in the query."
        
        try:
            relationships = []
            
            # Query for relationships involving each entity
            for entity in entities:
                query = """
                MATCH (e)-[r]-(n)
                WHERE toLower(e.name) CONTAINS toLower($entity) 
                OR toLower(n.name) CONTAINS toLower($entity)
                RETURN DISTINCT 
                    CASE 
                        WHEN toLower(startNode(r).name) CONTAINS toLower($entity)
                        THEN startNode(r).name + ' --[' + type(r) + ']--> ' + endNode(r).name
                        ELSE endNode(r).name + ' --[' + type(r) + ']--> ' + startNode(r).name
                    END as relationship
                LIMIT 5
                """
                
                result = self.graph.query(query, {"entity": entity})
                
                if result:
                    relationships.extend([row["relationship"] for row in result])
            
            if not relationships:
                return f"No relationships found for entities: {', '.join(entities)}"
            
            # Format the results
            context = "Found the following relationships:\n"
            context += "\n".join(f"- {rel}" for rel in relationships)
            return context
            
        except Exception as e:
            logger.error(f"Error querying graph database: {str(e)}")
            return f"Error retrieving relationships from graph database: {str(e)}"

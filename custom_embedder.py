import requests
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from typing import Union, List
from dspy.retrieve.neo4j_rm import Neo4jRM

class OllamaEmbedder:
    def __init__(self, model: str, api_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = api_url.rstrip('/')

    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        # Handle single string vs list of strings
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        all_embeddings = []
        for t in texts:
            response = requests.post(
                f"{self.api_url}/api/embeddings",
                json={"model": self.model, "prompt": t}
            )
            if response.status_code != 200:
                raise Exception(f"Error from Ollama API: {response.text}")
            
            embedding = response.json().get('embedding')
            if embedding is None:
                raise ValueError("No embedding returned from Ollama API")
            
            # Ensure embedding is a list of floats
            embedding = [float(x) for x in embedding]
            logging.debug(f"Embedding shape: {len(embedding)}, Type: {type(embedding[0])}")
            all_embeddings.append(embedding)
        
        # Convert to numpy array and ensure float32 type for compatibility
        embeddings = np.array(all_embeddings, dtype=np.float32)
        logging.debug(f"Final embeddings shape: {embeddings.shape}, Type: {embeddings.dtype}")
        
        # Always return a 2D array to ensure consistency
        return embeddings if len(texts) > 1 else embeddings.reshape(1, -1)

class OllamaNeo4jRM(Neo4jRM):
    def __init__(self, index_name: str, text_node_property: str, embedding_model: str, k: int = 5):
        embedder = OllamaEmbedder(model=embedding_model)
        super().__init__(
            index_name=index_name,
            text_node_property=text_node_property,
            k=k,
            embedding_function=embedder
        )

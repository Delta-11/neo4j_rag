# document_processor.py
import logging
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load and split a document into chunks."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Create a single document
            doc = Document(
                page_content=text,
                metadata={"source": file_path}
            )
            
            # Split the document
            docs = self.text_splitter.split_documents([doc])
            logging.info(f"Successfully loaded and split document into {len(docs)} chunks")
            return docs
        
        except Exception as e:
            logging.error(f"Error loading document {file_path}: {str(e)}")
            raise
class RAGFacade:
    def __init__(self):
        # Initialize components
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.neo4j_manager = Neo4jManager(self.config)
        self.embeddings_manager = EmbeddingsManager(self.config)
        self.rag = RAGPipeline(self.config)

    def process_documents(self, file_path: str):
        # Load and process document
        documents = self.doc_processor.load_document(file_path)
        logging.info(f"Loaded {len(documents)} document chunks")

        # Process documents to extract entities and relationships
        graph_documents = self.rag.process_documents(documents)
        logging.info(f"Processed {len(graph_documents)} documents for graph storage")

        # Add to Neo4j and create embeddings
        self.neo4j_manager.add_graph_documents(graph_documents)
        vector_retriever = self.embeddings_manager.create_vector_index()

        # Setup retrieval chain
        self.rag.setup_retrieval_chain(vector_retriever)

        return self.rag

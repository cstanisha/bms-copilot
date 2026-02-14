"""
Pinecone Vector Store Manager
Working imports for LangChain 0.3+ and Pinecone 5+
"""
from typing import List, Dict, Optional
import time

# Pinecone v5 imports
from pinecone import Pinecone, ServerlessSpec

# LangChain OpenAI
from langchain_openai import OpenAIEmbeddings

# LangChain Pinecone
from langchain_pinecone import PineconeVectorStore

# LangChain Core (for Documents)
from langchain_core.documents import Document


class PineconeManager:
    """Manages Pinecone vector database operations"""
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 3072,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize Pinecone manager
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: OpenAI embedding model name
            dimension: Embedding dimension
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model
        )
        
        self.index = None
        self.vector_store = None
    
    def initialize_index(self, delete_if_exists: bool = False) -> bool:
        """Initialize or create Pinecone index"""
        try:
            # List existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]
            
            if self.index_name in index_names:
                if delete_if_exists:
                    print(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                    time.sleep(1)
                else:
                    print(f"Using existing index: {self.index_name}")
                    self.index = self.pc.Index(self.index_name)
                    return True
            
            # Create new index
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            print(f"Index {self.index_name} created successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing index: {e}")
            return False
    
    def add_documents(
        self,
        documents: List[Document],
        namespace: str = "default"
    ) -> bool:
        """Add documents to Pinecone index"""
        try:
            if not self.index:
                self.initialize_index()
            
            # Reuse existing vector store or create new one
            if self.vector_store is None or self.vector_store.namespace != namespace:
                self.vector_store = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    index_name=self.index_name,
                    namespace=namespace
                )
            else:
                # Reuse existing connection, just add docs
                self.vector_store.add_documents(documents)
            
            print(f"✅ Added {len(documents)} documents to namespace '{namespace}'")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        namespace: str = "default",
        score_threshold: float = 0.0
    ) -> List[tuple]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                self.vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=namespace
                )
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_stats(self, namespace: str = "default") -> Dict:
        """Get index statistics"""
        try:
            if not self.index:
                self.index = self.pc.Index(self.index_name)
            
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'namespace_stats': stats.get('namespaces', {}),
                'dimension': stats.get('dimension', 0)
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace"""
        try:
            if not self.index:
                self.index = self.pc.Index(self.index_name)
            
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"Deleted namespace: {namespace}")
            return True
        except Exception as e:
            print(f"Error deleting namespace: {e}")
            return False
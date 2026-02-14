"""
Configuration Management for Fire Extinguisher RAG System
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4-turbo-preview"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))


class PineconeConfig(BaseModel):
    """Pinecone vector database configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    index_name: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "bms-agent"))
    dimension: int = 1536 
    metric: str = "cosine"
    cloud: str = "aws" 
    region: str = "us-east-1"  


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5")))
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35")))
    fetch_k: int = 20  # For MMR diversity


class ValidationConfig(BaseModel):
    """Validation configuration"""
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")))
    enable_hallucination_check: bool = Field(default_factory=lambda: os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true")
    min_source_coverage: float = 0.8  # 80% of claims must be sourced


class ChunkingConfig(BaseModel):
    """Text chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", ". ", " ", ""]
    length_function: str = "len"


class Config(BaseModel):
    """Main configuration class"""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate that all required configurations are set"""
        if not self.openai.api_key:
            return False, "OPENAI_API_KEY not set"
        if not self.pinecone.api_key:
            return False, "PINECONE_API_KEY not set"
        return True, None


# Global config instance
config = Config()

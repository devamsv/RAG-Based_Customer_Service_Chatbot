# -*- coding: utf-8 -*-
"""
Configuration settings for RAG Chatbot
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Settings
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    # Use a model that is available on the current Gemini API account.
    # (Your API key supports "gemini-flash-latest" / Gemini 2.5 models.)
    # You can override via env var `GEMINI_MODEL`.
    # Model available for generateContent with your API key.
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: int = 60
    
    # Document Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval Settings
    RETRIEVAL_K: int = 3  # Number of documents to retrieve
    RETRIEVAL_SEARCH_TYPE: str = "similarity"
    
    # Embedding Settings (REAL embeddings requirement)
    HF_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Dataset + Vector DB (persistent store)
    CUSTOMER_SUPPORT_CSV_PATH: str = "customer_support.csv"
    VECTOR_STORE_DIR: str = "vector_store"
    
    # UI Settings
    PAGE_TITLE: str = "Customer Support Chatbot"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    
    # Text Display Settings
    SOURCE_PREVIEW_LENGTH: int = 300

    # Metrics display controls
    SHOW_SOURCES: bool = True
    SHOW_EVALUATION_METRICS: bool = True
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.GOOGLE_API_KEY:
            print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
            return False
        return True


# Development configuration
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


# Production configuration
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


# Get configuration based on environment
ENV = os.getenv("ENVIRONMENT", "development").lower()
if ENV == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()

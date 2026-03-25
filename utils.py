# -*- coding: utf-8 -*-
"""
Utility functions for RAG Chatbot
"""

import os
import logging
from typing import Optional, List
from pathlib import Path


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Validate if file type is allowed
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed file extensions
        
    Returns:
        True if valid, False otherwise
    """
    file_extension = Path(filename).suffix.lstrip('.').lower()
    return file_extension in allowed_types


def get_file_extension(filename: str) -> str:
    """
    Get file extension without dot
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension in lowercase
    """
    return Path(filename).suffix.lstrip('.').lower()


def format_file_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {directory}")


def clean_temp_files(file_paths: List[str]) -> None:
    """
    Clean up temporary files safely
    
    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {file_path}: {str(e)}")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

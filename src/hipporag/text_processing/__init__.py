"""
独立的文本处理模块，包含多种文本分块策略。

本模块提供了一套完整的文本分块解决方案，包括：
- 语义分块：基于句子嵌入相似性的智能分块
- 固定分块：传统的固定长度分块
- 自适应分块：根据内容复杂度动态调整的分块
"""

from .base import BaseChunker, ChunkResult, ChunkingStrategy
from .config import ChunkingConfig, ThresholdType, EmbeddingProvider
from .semantic_chunker import SemanticChunker
from .utils import ChunkingUtils

__all__ = [
    'BaseChunker',
    'ChunkResult',
    'ChunkingStrategy', 
    'ChunkingConfig',
    'ThresholdType',
    'EmbeddingProvider',
    'SemanticChunker',
    'ChunkingUtils'
]

__version__ = '1.0.0'
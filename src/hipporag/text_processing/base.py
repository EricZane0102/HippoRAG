"""
分块器基类和接口定义。

定义了统一的分块器接口，支持不同的分块策略。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    SEMANTIC = "semantic"          # 语义分块
    FIXED = "fixed"               # 固定长度分块
    ADAPTIVE = "adaptive"         # 自适应分块
    RECURSIVE = "recursive"       # 递归分块


@dataclass
class ChunkResult:
    """分块结果数据类"""
    chunks: List[str]                    # 分块后的文本列表
    metadata: Dict[str, Any]             # 元数据信息
    strategy: ChunkingStrategy           # 使用的分块策略
    chunk_count: int                     # 块数量
    processing_time: float               # 处理时间（秒）
    
    def __post_init__(self):
        """自动计算块数量"""
        if self.chunk_count is None:
            self.chunk_count = len(self.chunks)
    
    def get_chunk_sizes(self) -> List[int]:
        """获取每个块的字符长度"""
        return [len(chunk) for chunk in self.chunks]
    
    def get_average_chunk_size(self) -> float:
        """获取平均块大小"""
        sizes = self.get_chunk_sizes()
        return sum(sizes) / len(sizes) if sizes else 0.0
    
    def summary(self) -> Dict[str, Any]:
        """返回分块结果摘要"""
        return {
            'strategy': self.strategy.value,
            'chunk_count': self.chunk_count,
            'processing_time': self.processing_time,
            'average_chunk_size': self.get_average_chunk_size(),
            'min_chunk_size': min(self.get_chunk_sizes()) if self.chunks else 0,
            'max_chunk_size': max(self.get_chunk_sizes()) if self.chunks else 0,
            'total_characters': sum(self.get_chunk_sizes())
        }


class BaseChunker(ABC):
    """
    分块器基类，定义了统一的分块接口。
    
    所有分块器实现都应继承此类并实现必要的抽象方法。
    """
    
    def __init__(self, strategy: ChunkingStrategy):
        """
        初始化分块器。
        
        Args:
            strategy: 分块策略
        """
        self.strategy = strategy
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> ChunkResult:
        """
        对单个文本进行分块。
        
        Args:
            text: 待分块的文本
            **kwargs: 额外的参数
            
        Returns:
            ChunkResult: 分块结果
        """
        pass
    
    def chunk_texts(self, texts: List[str], **kwargs) -> List[ChunkResult]:
        """
        对多个文本进行批量分块。
        
        Args:
            texts: 待分块的文本列表
            **kwargs: 额外的参数
            
        Returns:
            List[ChunkResult]: 分块结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.chunk_text(text, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"分块处理失败: {str(e)}")
                # 返回原文本作为单个块
                results.append(ChunkResult(
                    chunks=[text],
                    metadata={'error': str(e)},
                    strategy=self.strategy,
                    chunk_count=1,
                    processing_time=0.0
                ))
        return results
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                       text_key: str = 'content', **kwargs) -> List[Dict[str, Any]]:
        """
        对文档列表进行分块，保持文档结构。
        
        Args:
            documents: 文档列表，每个文档应包含text_key指定的文本字段
            text_key: 文档中文本内容的键名
            **kwargs: 额外的参数
            
        Returns:
            List[Dict[str, Any]]: 分块后的文档列表
        """
        chunked_documents = []
        
        for doc in documents:
            if text_key not in doc:
                self.logger.warning(f"文档中缺少文本字段 '{text_key}'")
                continue
                
            text = doc[text_key]
            chunk_result = self.chunk_text(text, **kwargs)
            
            # 为每个块创建新文档
            for i, chunk in enumerate(chunk_result.chunks):
                new_doc = doc.copy()
                new_doc[text_key] = chunk
                new_doc['chunk_index'] = i
                new_doc['original_doc_id'] = doc.get('id', f"doc_{id(doc)}")
                new_doc['chunking_strategy'] = self.strategy.value
                new_doc['chunk_metadata'] = chunk_result.metadata
                chunked_documents.append(new_doc)
        
        return chunked_documents
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        获取分块器配置。
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        pass
    
    def validate_input(self, text: str) -> bool:
        """
        验证输入文本。
        
        Args:
            text: 待验证的文本
            
        Returns:
            bool: 是否有效
        """
        if not isinstance(text, str):
            self.logger.error("输入必须是字符串类型")
            return False
        
        if not text.strip():
            self.logger.warning("输入文本为空")
            return False
        
        return True
    
    def _create_result(self, chunks: List[str], metadata: Dict[str, Any], 
                      processing_time: float) -> ChunkResult:
        """
        创建分块结果。
        
        Args:
            chunks: 分块列表
            metadata: 元数据
            processing_time: 处理时间
            
        Returns:
            ChunkResult: 分块结果
        """
        return ChunkResult(
            chunks=chunks,
            metadata=metadata,
            strategy=self.strategy,
            chunk_count=len(chunks),
            processing_time=processing_time
        )
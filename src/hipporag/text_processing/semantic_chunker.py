"""
语义分块器实现。

基于LangChain的SemanticChunker实现，提供智能的语义分块功能。
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
import re

try:
    from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .base import BaseChunker, ChunkResult, ChunkingStrategy
from .config import ChunkingConfig, SemanticChunkingConfig, ThresholdType, EmbeddingProvider

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    语义分块器，基于句子嵌入相似性进行智能分块。
    
    使用LangChain的SemanticChunker作为底层实现，支持多种嵌入模型和阈值策略。
    """
    
    def __init__(self, config: Union[ChunkingConfig, SemanticChunkingConfig]):
        """
        初始化语义分块器。
        
        Args:
            config: 分块配置
        """
        super().__init__(ChunkingStrategy.SEMANTIC)
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("语义分块需要安装langchain-experimental和相关依赖。请运行: pip install langchain-experimental langchain-openai")
        
        # 获取语义分块配置
        if isinstance(config, ChunkingConfig):
            self.config = config
            self.semantic_config = config.semantic_config
        else:
            self.config = ChunkingConfig(semantic_config=config)
            self.semantic_config = config
        
        # 初始化嵌入模型
        self.embedding_model = self._create_embedding_model()
        
        # 初始化LangChain语义分块器
        self.langchain_chunker = self._create_langchain_chunker()
        
        # 缓存
        self._embedding_cache = {} if self.semantic_config.enable_cache else None
        
        self.logger.info(f"语义分块器初始化完成，使用{self.semantic_config.embedding_provider.value}嵌入模型")
    
    def _create_embedding_model(self):
        """创建嵌入模型实例"""
        provider = self.semantic_config.embedding_provider
        model_name = self.semantic_config.embedding_model_name
        
        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=model_name,
                chunk_size=self.semantic_config.batch_size
            )
        elif provider == EmbeddingProvider.BGE_M3:
            return HuggingFaceEmbeddings(
                model_name=model_name or "BAAI/bge-m3",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise ValueError(f"不支持的嵌入模型提供商: {provider}")
    
    def _create_langchain_chunker(self) -> LangChainSemanticChunker:
        """创建LangChain语义分块器"""
        threshold_type_mapping = {
            ThresholdType.PERCENTILE: "percentile",
            ThresholdType.GRADIENT: "gradient", 
            ThresholdType.INTERQUARTILE: "interquartile",
            ThresholdType.STANDARD_DEVIATION: "standard_deviation"
        }
        
        return LangChainSemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type=threshold_type_mapping[self.semantic_config.threshold_type],
            breakpoint_threshold_amount=self.semantic_config.threshold_amount,
            number_of_chunks=None,  # 不限制块数量
            sentence_split_regex=r'(?<=[.!?])\s+',  # 句子分割正则表达式
            buffer_size=self.semantic_config.sentence_buffer_size
        )
    
    def chunk_text(self, text: str, **kwargs) -> ChunkResult:
        """
        对文本进行语义分块。
        
        Args:
            text: 待分块的文本
            **kwargs: 额外参数
            
        Returns:
            ChunkResult: 分块结果
        """
        start_time = time.time()
        
        # 验证输入
        if not self.validate_input(text):
            return self._create_result(
                chunks=[],
                metadata={'error': '无效输入'},
                processing_time=time.time() - start_time
            )
        
        try:
            self.logger.info(f"开始处理文本，长度: {len(text)}")
            
            # 预处理文本
            self.logger.info("开始预处理文本")
            processed_text = self._preprocess_text(text)
            self.logger.info(f"预处理完成，处理后长度: {len(processed_text)}")
            
            # 执行语义分块
            self.logger.info("开始执行语义分块")
            chunks = self._perform_semantic_chunking(processed_text)
            self.logger.info(f"语义分块完成，生成 {len(chunks)} 个块")
            
            # 后处理
            self.logger.info("开始后处理分块")
            chunks = self._postprocess_chunks(chunks)
            self.logger.info(f"后处理完成，最终 {len(chunks)} 个块")
            
            # 计算元数据
            metadata = self._compute_metadata(text, chunks)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"语义分块完成: {len(chunks)}个块, 耗时{processing_time:.2f}秒")
            
            return self._create_result(chunks, metadata, processing_time)
            
        except Exception as e:
            self.logger.error(f"语义分块失败: {str(e)}")
            processing_time = time.time() - start_time
            return self._create_result(
                chunks=[text],  # 失败时返回原文本
                metadata={'error': str(e), 'fallback': True},
                processing_time=processing_time
            )
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if self.config.strip_whitespace:
            # 清理多余的空白字符
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _perform_semantic_chunking(self, text: str) -> List[str]:
        """执行语义分块"""
        self.logger.info(f"开始语义分块，文本长度: {len(text)}")
        self.logger.info(f"分块配置 - 阈值类型: {self.semantic_config.threshold_type}, 阈值: {self.semantic_config.threshold_amount}")
        
        # 使用LangChain的语义分块器
        documents = self.langchain_chunker.create_documents([text])
        
        # 提取文本内容
        chunks = [doc.page_content for doc in documents]
        
        self.logger.info(f"语义分块结果: {len(chunks)}个块")
        for i, chunk in enumerate(chunks):
            self.logger.info(f"块 {i+1}: {len(chunk)} 字符")
        
        return chunks
    
    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """后处理分块结果"""
        processed_chunks = []
        
        for chunk in chunks:
            # 清理空白字符
            if self.config.strip_whitespace:
                chunk = chunk.strip()
            
            # 检查最小长度
            if len(chunk) < self.semantic_config.min_chunk_size:
                if processed_chunks and len(processed_chunks[-1]) < self.semantic_config.max_chunk_size:
                    # 合并到前一个块
                    processed_chunks[-1] += " " + chunk
                else:
                    processed_chunks.append(chunk)
            else:
                processed_chunks.append(chunk)
        
        # 移除空块
        if self.config.remove_empty_chunks:
            processed_chunks = [chunk for chunk in processed_chunks if chunk.strip()]
        
        return processed_chunks
    
    def _compute_metadata(self, original_text: str, chunks: List[str]) -> Dict[str, Any]:
        """计算分块元数据"""
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        metadata = {
            'original_length': len(original_text),
            'chunk_count': len(chunks),
            'chunk_sizes': chunk_sizes,
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'compression_ratio': sum(chunk_sizes) / len(original_text) if original_text else 0,
            'embedding_provider': self.semantic_config.embedding_provider.value,
            'embedding_model': self.semantic_config.embedding_model_name,
            'threshold_type': self.semantic_config.threshold_type.value,
            'threshold_amount': self.semantic_config.threshold_amount
        }
        
        return metadata
    
    def get_config(self) -> Dict[str, Any]:
        """获取分块器配置"""
        return self.config.to_dict()
    
    def batch_chunk_texts(self, texts: List[str], **kwargs) -> List[ChunkResult]:
        """
        批量处理多个文本的语义分块。
        
        Args:
            texts: 文本列表
            **kwargs: 额外参数
            
        Returns:
            List[ChunkResult]: 分块结果列表
        """
        results = []
        
        self.logger.info(f"开始批量语义分块，共{len(texts)}个文本")
        
        for i, text in enumerate(texts):
            try:
                self.logger.info(f"正在处理第 {i+1}/{len(texts)} 个文本，长度: {len(text)}")
                result = self.chunk_text(text, **kwargs)
                results.append(result)
                self.logger.info(f"第 {i+1} 个文本处理完成，生成 {result.chunk_count} 个块")
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(texts)} 个文本")
                    
            except Exception as e:
                self.logger.error(f"处理第{i+1}个文本时出错: {str(e)}")
                results.append(ChunkResult(
                    chunks=[text],
                    metadata={'error': str(e)},
                    strategy=self.strategy,
                    chunk_count=1,
                    processing_time=0.0
                ))
        
        self.logger.info(f"批量语义分块完成，共处理{len(texts)}个文本")
        return results
    
    def update_config(self, new_config: Union[ChunkingConfig, SemanticChunkingConfig]):
        """
        更新分块器配置。
        
        Args:
            new_config: 新的配置
        """
        if isinstance(new_config, ChunkingConfig):
            self.config = new_config
            self.semantic_config = new_config.semantic_config
        else:
            self.config.semantic_config = new_config
            self.semantic_config = new_config
        
        # 重新创建嵌入模型和分块器
        self.embedding_model = self._create_embedding_model()
        self.langchain_chunker = self._create_langchain_chunker()
        
        self.logger.info("分块器配置已更新")


class SemanticChunkerFactory:
    """语义分块器工厂类"""
    
    @staticmethod
    def create_default() -> SemanticChunker:
        """创建默认配置的语义分块器"""
        from .config import DEFAULT_SEMANTIC_CONFIG
        return SemanticChunker(DEFAULT_SEMANTIC_CONFIG)
    
    @staticmethod
    def create_high_precision() -> SemanticChunker:
        """创建高精度配置的语义分块器"""
        from .config import HIGH_PRECISION_SEMANTIC_CONFIG
        return SemanticChunker(HIGH_PRECISION_SEMANTIC_CONFIG)
    
    @staticmethod
    def create_with_bge_m3() -> SemanticChunker:
        """创建使用BGE-M3嵌入模型的分块器"""
        config = ChunkingConfig()
        config.semantic_config.embedding_provider = EmbeddingProvider.BGE_M3
        config.semantic_config.embedding_model_name = "BAAI/bge-m3"
        return SemanticChunker(config)
    
    @staticmethod
    def create_custom(embedding_provider: EmbeddingProvider,
                     embedding_model: str,
                     threshold_type: ThresholdType = ThresholdType.PERCENTILE,
                     threshold_amount: float = 95.0) -> SemanticChunker:
        """
        创建自定义配置的语义分块器。
        
        Args:
            embedding_provider: 嵌入模型提供商
            embedding_model: 嵌入模型名称
            threshold_type: 阈值类型
            threshold_amount: 阈值数值
            
        Returns:
            SemanticChunker: 配置好的分块器
        """
        config = ChunkingConfig()
        config.semantic_config.embedding_provider = embedding_provider
        config.semantic_config.embedding_model_name = embedding_model
        config.semantic_config.threshold_type = threshold_type
        config.semantic_config.threshold_amount = threshold_amount
        
        return SemanticChunker(config)
"""
分块配置类定义。

提供了灵活的配置选项来控制不同分块策略的行为。
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union
from enum import Enum

from .base import ChunkingStrategy


class ThresholdType(str, Enum):
    """语义分块阈值类型"""
    PERCENTILE = "percentile"        # 百分位阈值
    GRADIENT = "gradient"           # 梯度阈值
    INTERQUARTILE = "interquartile" # 四分位数阈值
    STANDARD_DEVIATION = "standard_deviation"  # 标准差阈值


class EmbeddingProvider(str, Enum):
    """嵌入模型提供商"""
    OPENAI = "openai"
    BGE_M3 = "bge_m3"
    NVEMBED_V2 = "nvembed_v2"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class SemanticChunkingConfig:
    """语义分块配置"""
    
    # 嵌入模型配置
    embedding_provider: EmbeddingProvider = field(
        default=EmbeddingProvider.OPENAI,
        metadata={"help": "嵌入模型提供商"}
    )
    
    embedding_model_name: str = field(
        default="text-embedding-ada-002",
        metadata={"help": "嵌入模型名称"}
    )
    
    # 阈值配置
    threshold_type: ThresholdType = field(
        default=ThresholdType.PERCENTILE,
        metadata={"help": "阈值计算方法"}
    )
    
    threshold_amount: float = field(
        default=95.0,
        metadata={"help": "阈值数值，具体含义取决于threshold_type"}
    )
    
    # 句子分组配置
    sentence_buffer_size: int = field(
        default=1,
        metadata={"help": "句子缓冲区大小，用于分组"}
    )
    
    # 处理配置
    max_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "最大块大小限制（字符数）"}
    )
    
    min_chunk_size: int = field(
        default=50,
        metadata={"help": "最小块大小（字符数）"}
    )
    
    # 性能配置
    batch_size: int = field(
        default=16,
        metadata={"help": "批量处理大小"}
    )
    
    enable_cache: bool = field(
        default=True,
        metadata={"help": "是否启用嵌入缓存"}
    )


@dataclass
class FixedChunkingConfig:
    """固定长度分块配置"""
    
    chunk_size: int = field(
        default=1000,
        metadata={"help": "固定块大小（字符数）"}
    )
    
    chunk_overlap: int = field(
        default=100,
        metadata={"help": "块之间的重叠字符数"}
    )
    
    separator: str = field(
        default="\n\n",
        metadata={"help": "分隔符"}
    )
    
    keep_separator: bool = field(
        default=True,
        metadata={"help": "是否保留分隔符"}
    )


@dataclass
class AdaptiveChunkingConfig:
    """自适应分块配置"""
    
    base_chunk_size: int = field(
        default=800,
        metadata={"help": "基础块大小"}
    )
    
    min_chunk_size: int = field(
        default=200,
        metadata={"help": "最小块大小"}
    )
    
    max_chunk_size: int = field(
        default=2000,
        metadata={"help": "最大块大小"}
    )
    
    complexity_threshold: float = field(
        default=0.5,
        metadata={"help": "复杂度阈值"}
    )
    
    adaptation_factor: float = field(
        default=0.3,
        metadata={"help": "适应系数"}
    )


@dataclass 
class ChunkingConfig:
    """统一的分块配置类"""
    
    # 通用配置
    strategy: ChunkingStrategy = field(
        default=ChunkingStrategy.SEMANTIC,
        metadata={"help": "分块策略"}
    )
    
    # 文本预处理
    strip_whitespace: bool = field(
        default=True,
        metadata={"help": "是否清理空白字符"}
    )
    
    remove_empty_chunks: bool = field(
        default=True,
        metadata={"help": "是否移除空块"}
    )
    
    # 特定策略配置
    semantic_config: SemanticChunkingConfig = field(
        default_factory=SemanticChunkingConfig,
        metadata={"help": "语义分块配置"}
    )
    
    fixed_config: FixedChunkingConfig = field(
        default_factory=FixedChunkingConfig,
        metadata={"help": "固定分块配置"}
    )
    
    adaptive_config: AdaptiveChunkingConfig = field(
        default_factory=AdaptiveChunkingConfig,
        metadata={"help": "自适应分块配置"}
    )
    
    # 日志和调试
    enable_debug: bool = field(
        default=False,
        metadata={"help": "是否启用调试模式"}
    )
    
    log_level: str = field(
        default="INFO",
        metadata={"help": "日志级别"}
    )
    
    def get_strategy_config(self) -> Union[SemanticChunkingConfig, FixedChunkingConfig, AdaptiveChunkingConfig]:
        """根据当前策略获取对应的配置"""
        if self.strategy == ChunkingStrategy.SEMANTIC:
            return self.semantic_config
        elif self.strategy == ChunkingStrategy.FIXED:
            return self.fixed_config  
        elif self.strategy == ChunkingStrategy.ADAPTIVE:
            return self.adaptive_config
        else:
            raise ValueError(f"不支持的分块策略: {self.strategy}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'strategy': self.strategy.value,
            'strip_whitespace': self.strip_whitespace,
            'remove_empty_chunks': self.remove_empty_chunks,
            'enable_debug': self.enable_debug,
            'log_level': self.log_level
        }
        
        if self.strategy == ChunkingStrategy.SEMANTIC:
            result['semantic_config'] = {
                'embedding_provider': self.semantic_config.embedding_provider.value,
                'embedding_model_name': self.semantic_config.embedding_model_name,
                'threshold_type': self.semantic_config.threshold_type.value,
                'threshold_amount': self.semantic_config.threshold_amount,
                'sentence_buffer_size': self.semantic_config.sentence_buffer_size,
                'max_chunk_size': self.semantic_config.max_chunk_size,
                'min_chunk_size': self.semantic_config.min_chunk_size,
                'batch_size': self.semantic_config.batch_size,
                'enable_cache': self.semantic_config.enable_cache
            }
        elif self.strategy == ChunkingStrategy.FIXED:
            result['fixed_config'] = {
                'chunk_size': self.fixed_config.chunk_size,
                'chunk_overlap': self.fixed_config.chunk_overlap,
                'separator': self.fixed_config.separator,
                'keep_separator': self.fixed_config.keep_separator
            }
        elif self.strategy == ChunkingStrategy.ADAPTIVE:
            result['adaptive_config'] = {
                'base_chunk_size': self.adaptive_config.base_chunk_size,
                'min_chunk_size': self.adaptive_config.min_chunk_size,
                'max_chunk_size': self.adaptive_config.max_chunk_size,
                'complexity_threshold': self.adaptive_config.complexity_threshold,
                'adaptation_factor': self.adaptive_config.adaptation_factor
            }
        
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChunkingConfig':
        """从字典创建配置对象"""
        strategy = ChunkingStrategy(config_dict.get('strategy', ChunkingStrategy.SEMANTIC))
        
        config = cls(
            strategy=strategy,
            strip_whitespace=config_dict.get('strip_whitespace', True),
            remove_empty_chunks=config_dict.get('remove_empty_chunks', True),
            enable_debug=config_dict.get('enable_debug', False),
            log_level=config_dict.get('log_level', 'INFO')
        )
        
        # 根据策略设置对应的配置
        if strategy == ChunkingStrategy.SEMANTIC and 'semantic_config' in config_dict:
            semantic_dict = config_dict['semantic_config']
            config.semantic_config = SemanticChunkingConfig(
                embedding_provider=EmbeddingProvider(semantic_dict.get('embedding_provider', 'openai')),
                embedding_model_name=semantic_dict.get('embedding_model_name', 'text-embedding-ada-002'),
                threshold_type=ThresholdType(semantic_dict.get('threshold_type', 'percentile')),
                threshold_amount=semantic_dict.get('threshold_amount', 95.0),
                sentence_buffer_size=semantic_dict.get('sentence_buffer_size', 1),
                max_chunk_size=semantic_dict.get('max_chunk_size'),
                min_chunk_size=semantic_dict.get('min_chunk_size', 50),
                batch_size=semantic_dict.get('batch_size', 16),
                enable_cache=semantic_dict.get('enable_cache', True)
            )
        
        return config


# 预定义配置
DEFAULT_SEMANTIC_CONFIG = ChunkingConfig(
    strategy=ChunkingStrategy.SEMANTIC,
    semantic_config=SemanticChunkingConfig(
        embedding_provider=EmbeddingProvider.OPENAI,
        embedding_model_name="text-embedding-ada-002",
        threshold_type=ThresholdType.PERCENTILE,
        threshold_amount=95.0
    )
)

DEFAULT_FIXED_CONFIG = ChunkingConfig(
    strategy=ChunkingStrategy.FIXED,
    fixed_config=FixedChunkingConfig(
        chunk_size=1000,
        chunk_overlap=100
    )
)

HIGH_PRECISION_SEMANTIC_CONFIG = ChunkingConfig(
    strategy=ChunkingStrategy.SEMANTIC,
    semantic_config=SemanticChunkingConfig(
        embedding_provider=EmbeddingProvider.OPENAI,
        threshold_type=ThresholdType.PERCENTILE,
        threshold_amount=98.0,  # 更高的精度
        min_chunk_size=100
    )
)
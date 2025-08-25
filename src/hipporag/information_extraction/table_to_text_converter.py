import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("langchain-text-splitters未安装，将使用简单文本分割")
    RecursiveCharacterTextSplitter = None

from ..prompts.prompt_template_manager import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import compute_mdhash_id
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class TableConversionInfo(TypedDict, total=False):
    content: str
    context: str
    content_type: str
    chunk_id: str


@dataclass
class TableTextOutput:
    """表格转文本输出结果"""
    chunk_id: str
    original_table: str
    text_description: str
    text_chunks: List[str]
    chunk_ids: List[str]
    metadata: Dict[str, Any]


class TextChunker:
    """基于LangChain的文本分段器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if RecursiveCharacterTextSplitter is not None:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["。", "！", "？", "；", ".", "!", "?", "\n", " ", ""]
            )
        else:
            self.splitter = None
        
    def chunk_text(self, text: str, base_chunk_id: str) -> Tuple[List[str], List[str]]:
        """
        将文本分割成chunk
        
        Args:
            text: 要分割的文本
            base_chunk_id: 基础chunk ID
            
        Returns:
            Tuple[chunks, chunk_ids]
        """
        if len(text) <= self.chunk_size:
            return [text], [f"{base_chunk_id}_chunk_0"]
            
        if self.splitter is not None:
            # 使用LangChain的分割器
            chunks = self.splitter.split_text(text)
        else:
            # 简单fallback分割
            chunks = self._simple_split(text)
        
        # 生成chunk IDs
        chunk_ids = [f"{base_chunk_id}_chunk_{i}" for i in range(len(chunks))]
        
        return chunks, chunk_ids
    
    def _simple_split(self, text: str) -> List[str]:
        """简单的文本分割fallback方法"""
        sentences = re.split(r'[。！？；]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 添加句号
            if not sentence.endswith(('。', '！', '？', '；')):
                sentence += '。'
                
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class TableToTextConverter:
    """
    表格转文本描述转换器
    """
    
    def __init__(self, llm_model: CacheOpenAI, chunk_size: int = 500, 
                 overlap: int = 50, detail_level: str = "detailed"):
        self.llm_model = llm_model
        self.detail_level = detail_level
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        
        # 初始化提示模板管理器
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        
    def convert_table_to_text(self, table_html: str, table_context: str = "", 
                             chunk_id: Optional[str] = None) -> TableTextOutput:
        """
        将HTML表格转换为文本描述
        
        Args:
            table_html: HTML格式的表格内容
            table_context: 表格的上下文信息
            chunk_id: 可选的chunk ID
            
        Returns:
            TableTextOutput: 包含转换结果的数据结构
        """
        
        # 生成chunk_id（若未传入）
        if not chunk_id:
            chunk_id = compute_mdhash_id(table_html, prefix="table-")
            
        logger.info(f"转换表格 {chunk_id}")
        
        # 转换开始时间
        import time
        start_time = time.time()
        
        try:
            # 构建输入消息
            conversion_input = self.prompt_template_manager.render(
                name='table_to_text',
                context=table_context if table_context else "无额外上下文",
                table_html=table_html
            )
            
            # 调用LLM进行文本转换
            logger.info(f"正在将表格转换为文本描述...")
            response_message, _, _ = self.llm_model.infer(
                messages=conversion_input,
                temperature=0.3,
                response_format=None  # 确保使用文本格式而不是JSON格式
            )
            
            text_description = response_message.strip()
            
            # 清理LLM输出中的markdown格式和特殊字符
            text_description = self._clean_llm_output(text_description)
            
            conversion_time = time.time() - start_time
            
            # 将文本描述分割成chunks
            logger.debug("开始文本分段")
            chunk_start_time = time.time()
            text_chunks, chunk_ids = self.chunker.chunk_text(text_description, chunk_id)
            chunk_time = time.time() - chunk_start_time
            
            # 记录详细统计信息
            logger.info(f"表格转换完成: 生成 {len(text_chunks)} 个文本段落, 耗时 {conversion_time:.2f}s")
            logger.debug(f"文本分段耗时: {chunk_time:.2f}s")
            
            return TableTextOutput(
                chunk_id=chunk_id,
                original_table=table_html,
                text_description=text_description,
                text_chunks=text_chunks,
                chunk_ids=chunk_ids,
                metadata={
                    'detail_level': self.detail_level,
                    'timing': {
                        'conversion_time': conversion_time,
                        'chunk_time': chunk_time,
                        'total_time': conversion_time + chunk_time
                    }
                }
            )
            
        except Exception as e:
            conversion_time = time.time() - start_time
            logger.error(f"表格转文本失败 (耗时 {conversion_time:.2f}s): {e}")
            # 返回错误结果
            return TableTextOutput(
                chunk_id=chunk_id,
                original_table=table_html,
                text_description=f"转换失败: {str(e)}",
                text_chunks=[],
                chunk_ids=[],
                metadata={
                    'error': str(e),
                    'conversion_time': conversion_time
                }
            )
    
    
    def _clean_llm_output(self, text: str) -> str:
        """
        清理LLM输出中的markdown格式和特殊字符
        
        Args:
            text: 原始LLM输出
            
        Returns:
            清理后的文本
        """
        # 移除markdown标题
        text = re.sub(r'#{1,6}\s*', '', text)
        
        # 移除分隔符
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        
        # 移除多余的换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除行首的空格
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        # 确保句子结尾有标点
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('。', '！', '？', '；', ':', '：')):
                line += '。'
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    
    
    def convert_batch_tables(self, table_infos: List[TableConversionInfo], 
                           max_workers: Optional[int] = None, 
                           show_progress: bool = True) -> List[TableTextOutput]:
        """
        批量转换多个表格
        
        Args:
            table_infos: 表格信息列表
            max_workers: 最大并发数
            show_progress: 是否显示进度条
            
        Returns:
            所有表格的转换结果
        """
        if max_workers is None or max_workers <= 0:
            max_workers = 4

        total = len(table_infos)
        logger.info(f"批量转换表格，共 {total} 个，最大并发 {max_workers}")

        results: List[Optional[TableTextOutput]] = [None] * total
        
        # 统计信息
        import time
        start_time = time.time()
        success_count = 0
        error_count = 0
        total_chunks = 0
        total_conversion_time = 0

        def _process_one(idx: int, info: TableConversionInfo) -> Tuple[int, TableTextOutput]:
            res = self.convert_table_to_text(
                table_html=info['content'],
                table_context=info.get('context', ''),
                chunk_id=info.get('chunk_id')
            )
            return idx, res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, i, info): i for i, info in enumerate(table_infos)}
            pbar = tqdm(total=total, desc="转换表格为文本", unit="table", disable=not show_progress)
            
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                    
                    # 更新统计信息
                    if res.text_chunks:
                        success_count += 1
                        total_chunks += len(res.text_chunks)
                        if 'timing' in res.metadata:
                            total_conversion_time += res.metadata['timing'].get('total_time', 0)
                    else:
                        error_count += 1
                        
                except Exception as e:
                    idx = futures[future]
                    error_count += 1
                    logger.warning(f"第 {idx+1} 个表格转换失败: {e}")
                    # 创建错误结果
                    fallback_chunk_id = table_infos[idx].get('chunk_id') or compute_mdhash_id(
                        table_infos[idx]['content'], prefix="table-")
                    results[idx] = TableTextOutput(
                        chunk_id=fallback_chunk_id,
                        original_table=table_infos[idx]['content'],
                        text_description=f"转换失败: {str(e)}",
                        text_chunks=[],
                        chunk_ids=[],
                        metadata={'error': str(e)}
                    )
                finally:
                    pbar.update(1)
            pbar.close()

        # 确保所有结果都有效
        finalized_results: List[TableTextOutput] = []
        for i, r in enumerate(results):
            if r is None:
                fallback_chunk_id = table_infos[i].get('chunk_id') or compute_mdhash_id(
                    table_infos[i]['content'], prefix="table-")
                r = TableTextOutput(
                    chunk_id=fallback_chunk_id,
                    original_table=table_infos[i]['content'],
                    text_description="转换失败: Unknown Error",
                    text_chunks=[],
                    chunk_ids=[],
                    metadata={'error': 'Unknown Error'}
                )
                error_count += 1
            finalized_results.append(r)

        # 计算和记录统计信息
        total_time = time.time() - start_time
        avg_time_per_table = total_conversion_time / success_count if success_count > 0 else 0
        avg_chunks_per_table = total_chunks / success_count if success_count > 0 else 0
        
        logger.info("批量表格转换完成 - 统计信息:")
        logger.info(f"  总耗时: {total_time:.2f}s")
        logger.info(f"  成功转换: {success_count}/{total} ({success_count/total*100:.1f}%)")
        logger.info(f"  转换失败: {error_count}/{total} ({error_count/total*100:.1f}%)")
        logger.info(f"  生成文本段落: {total_chunks} 个")
        logger.info(f"  平均转换时间: {avg_time_per_table:.2f}s/表格")
        logger.info(f"  平均段落数: {avg_chunks_per_table:.1f} 段/表格")

        return finalized_results
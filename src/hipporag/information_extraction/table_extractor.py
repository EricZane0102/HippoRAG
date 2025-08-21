import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("BeautifulSoup4 未安装，表格处理功能将受限")
    BeautifulSoup = None

from ..prompts.prompt_template_manager import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import TripleRawOutput, compute_mdhash_id
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class TableTripleInfo(TypedDict, total=False):
    content: str
    context: str
    content_type: str
    chunk_id: str


@dataclass
class TableLLMInput:
    table_id: str
    input_message: List[Dict]


class TableTripleExtractor:
    """
    使用Few-Shot Prompt Engineering从HTML表格提取知识三元组的提取器
    """
    
    def __init__(self, llm_model: CacheOpenAI):
        # 初始化提示模板管理器
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        
    def extract_triples_from_table(self, table_html: str, table_context: str = "", chunk_id: Optional[str] = None) -> TripleRawOutput:
        """
        从HTML表格中提取三元组
        
        Args:
            table_html: HTML格式的表格内容
            table_context: 表格的上下文信息（如标题、来源等）
            
        Returns:
            TripleRawOutput: 包含提取结果的数据结构
        """
        
        # 简化HTML表格
        logger.debug("开始简化表格HTML")
        simplified_table = self._simplify_html_table(table_html)
        
        # 构建输入消息
        table_extraction_input = self.prompt_template_manager.render(
            name='table_extraction',
            context=table_context if table_context else "无额外上下文",
            table_html=simplified_table
        )
        
        raw_response = ""
        metadata = {}
        triples = []
        
        try:
            # 预先解析表格文本（用于后处理验证，避免重复解析）
            logger.debug("解析表格文本用于后处理")
            try:
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(table_html, 'html.parser')
                    table_text = soup.get_text().lower()
                else:
                    table_text = re.sub(r'<[^>]+>', ' ', table_html).lower()
            except Exception:
                table_text = table_html.lower()

            # 调用LLM进行三元组提取
            logger.info(f"正在从表格提取三元组...")
            response_message, metadata, _ = self.llm_model.infer(
                messages=table_extraction_input,
                temperature=0.1
            )
            raw_response = response_message

            # 解析LLM输出的三元组
            logger.debug("开始解析LLM输出三元组")
            triples = self._parse_llm_triples(raw_response)

            # 后处理和验证（使用已解析的 table_text，避免重复解析HTML）
            logger.debug("开始三元组后处理")
            triples = self._postprocess_triples(triples, table_html, table_text=table_text)

            logger.info(f"成功提取 {len(triples)} 个三元组")

        except Exception as e:
            logger.error(f"表格三元组提取失败: {e}")
            raw_response = f"Error: {str(e)}"

        # 确保有 chunk_id（若未传入则基于内容生成）
        if not chunk_id:
            chunk_id = compute_mdhash_id(table_html, prefix="chunk-")

        return TripleRawOutput(
            chunk_id=chunk_id,
            response=raw_response,
            triples=triples,
            metadata=metadata
        )
    
    def _simplify_html_table(self, html_table: str) -> str:
        """
        简化HTML表格，只保留结构信息，去除不必要的属性
        
        Args:
            html_table: 原始HTML表格
            
        Returns:
            简化后的HTML表格
        """
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup未安装，返回原始表格HTML")
            return html_table
            
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            
            # 对所有标签只保留rowspan和colspan属性
            for tag in soup.find_all(True):
                attrs_to_keep = {}
                if 'rowspan' in tag.attrs:
                    attrs_to_keep['rowspan'] = tag.attrs['rowspan']
                if 'colspan' in tag.attrs:
                    attrs_to_keep['colspan'] = tag.attrs['colspan']
                tag.attrs = attrs_to_keep
            
            # 清理文本内容，移除多余空白
            for tag in soup.find_all(text=True):
                if tag.strip():
                    tag.replace_with(tag.strip())
                else:
                    tag.extract()
                    
            return str(soup)
            
        except Exception as e:
            logger.warning(f"HTML表格简化失败: {e}，使用原始表格")
            return html_table
    
    def _parse_llm_triples(self, response: str) -> List[Tuple[str, str, str]]:
        """
        解析LLM输出的三元组
        
        Args:
            response: LLM的原始响应
            
        Returns:
            三元组列表
        """
        triples = []
        
        # 按行分割响应
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#') or line.startswith('//'):
                continue
                
            # 尝试解析三元组格式：(主体, 关系, 客体)
            match = re.match(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                subject, predicate, obj = match.groups()
                # 清理空白和引号
                subject = subject.strip().strip('"\'')
                predicate = predicate.strip().strip('"\'')
                obj = obj.strip().strip('"\'')
                
                if subject and predicate and obj:
                    triples.append((subject, predicate, obj))
                    
        return triples
    
    def _postprocess_triples(self, triples: List[Tuple[str, str, str]], original_table: str, table_text: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """
        后处理三元组：验证、去重、过滤
        
        Args:
            triples: 原始三元组列表
            original_table: 原始表格HTML
            
        Returns:
            处理后的三元组列表
        """
        
        # 提取原表格的文本内容用于验证（允许外部传入，避免重复解析）
        if table_text is None:
            try:
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(original_table, 'html.parser')
                    table_text = soup.get_text().lower()
                else:
                    # 简单的HTML标签清理
                    table_text = re.sub(r'<[^>]+>', ' ', original_table).lower()
            except Exception:
                table_text = original_table.lower()
        
        processed_triples = []
        seen_triples = set()
        
        for subject, predicate, obj in triples:
            # 创建三元组的唯一标识
            triple_key = (subject.lower(), predicate.lower(), obj.lower())
            
            # 去重
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            
            # 验证：检查实体是否来源于原表格
            # 虚拟交叉节点总是保留
            is_intersection_node = ('交叉点' in subject or '交叉点' in obj)
            
            # 检查实体是否在原表格中出现
            subject_in_table = (subject.lower() in table_text or 
                              any(word in table_text for word in subject.lower().split() if len(word) > 1))
            obj_in_table = (obj.lower() in table_text or 
                          any(word in table_text for word in obj.lower().split() if len(word) > 1))
            
            # 保留条件：虚拟交叉节点 或 至少一个实体在原表格中
            if is_intersection_node or subject_in_table or obj_in_table:
                processed_triples.append((subject, predicate, obj))
            else:
                logger.debug(f"过滤掉不相关的三元组: ({subject}, {predicate}, {obj})")
        
        logger.info(f"后处理完成：保留 {len(processed_triples)} 个三元组")
        return processed_triples
    
    def extract_batch_tables(self, table_infos: List[TableTripleInfo], max_workers: Optional[int] = None, show_progress: bool = True) -> List[TripleRawOutput]:
        """
        批量处理多个表格
        
        Args:
            table_infos: 表格信息列表
            
        Returns:
            所有表格的三元组提取结果
        """
        if max_workers is None or max_workers <= 0:
            max_workers = 4

        total = len(table_infos)
        logger.info(f"批量处理表格，共 {total} 个，最大并发 {max_workers}")

        results: List[Optional[TripleRawOutput]] = [None] * total

        def _process_one(idx: int, info: TableTripleInfo) -> Tuple[int, TripleRawOutput]:
            res = self.extract_triples_from_table(
                table_html=info['content'],
                table_context=info.get('context', ''),
                chunk_id=info.get('chunk_id')
            )
            return idx, res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, i, info): i for i, info in enumerate(table_infos)}
            pbar = tqdm(total=total, desc="提取表格三元组", unit="tab", disable=not show_progress)
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"第 {idx+1} 个表格处理失败: {e}")
                    # 失败时返回空结果（尽量不中断整体流程）
                    fallback_chunk_id = table_infos[idx].get('chunk_id') or compute_mdhash_id(table_infos[idx]['content'], prefix="chunk-")
                    results[idx] = TripleRawOutput(
                        chunk_id=fallback_chunk_id,
                        response=f"提取失败: {str(e)}",
                        triples=[],
                        metadata={}
                    )
                finally:
                    pbar.update(1)
            pbar.close()

        # 类型断言与填充
        finalized_results: List[TripleRawOutput] = []
        for i, r in enumerate(results):
            if r is None:
                fallback_chunk_id = table_infos[i].get('chunk_id') or compute_mdhash_id(table_infos[i]['content'], prefix="chunk-")
                r = TripleRawOutput(
                    chunk_id=fallback_chunk_id,
                    response="提取失败: Unknown Error",
                    triples=[],
                    metadata={}
                )
            finalized_results.append(r)

        return finalized_results
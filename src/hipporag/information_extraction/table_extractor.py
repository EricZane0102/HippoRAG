import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional, Set
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
        
        # 提取表格元数据
        logger.debug("提取表格元数据")
        table_metadata = self._extract_table_metadata(table_html)
        
        # 简化HTML表格
        logger.debug("开始简化表格HTML")
        simplified_table = self._simplify_html_table(table_html)
        
        # 构建增强的上下文信息
        enhanced_context = self._build_enhanced_context(table_context, table_metadata)
        
        # 构建输入消息
        table_extraction_input = self.prompt_template_manager.render(
            name='table_extraction',
            context=enhanced_context,
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
            triples = self._postprocess_triples_enhanced(triples, table_html, table_text, table_metadata)

            logger.info(f"成功提取 {len(triples)} 个三元组")

        except Exception as e:
            logger.error(f"表格三元组提取失败: {e}")
            raw_response = f"Error: {str(e)}"

        # 确保有 chunk_id（若未传入则基于内容生成）
        if not chunk_id:
            chunk_id = compute_mdhash_id(table_html, prefix="chunk-")

        # 将表格元数据添加到最终metadata中
        metadata.update({
            'table_metadata': table_metadata,
            'enhanced_context': enhanced_context
        })

        return TripleRawOutput(
            chunk_id=chunk_id,
            response=raw_response,
            triples=triples,
            metadata=metadata
        )
        
    def _build_enhanced_context(self, original_context: str, metadata: Dict[str, Any]) -> str:
        """
        构建增强的上下文信息
        
        Args:
            original_context: 原始上下文
            metadata: 表格元数据
            
        Returns:
            增强的上下文字符串
        """
        enhanced_parts = []
        
        if original_context:
            enhanced_parts.append(f"原始上下文：{original_context}")
            
        # 添加表格结构信息
        if metadata['row_count'] > 0:
            enhanced_parts.append(f"表格规模：{metadata['row_count']}行 x {metadata['col_count']}列")
            
        if metadata['headers']:
            enhanced_parts.append(f"表头信息：{', '.join(metadata['headers'])}")
            
        enhanced_parts.append(f"表格类型：{metadata['structure_type']}")
        
        return "；".join(enhanced_parts) if enhanced_parts else "无额外上下文"
    
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
        解析LLM输出的三元组，支持多种格式和结构化信息
        
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
        
    def _extract_table_metadata(self, table_html: str) -> Dict[str, Any]:
        """
        提取表格的元数据信息，用于增强召回效果
        
        Args:
            table_html: HTML表格内容
            
        Returns:
            表格元数据字典
        """
        metadata = {
            'row_count': 0,
            'col_count': 0,
            'headers': [],
            'cell_values': [],
            'structure_type': 'simple'  # simple, hierarchical, complex
        }
        
        if BeautifulSoup is None:
            return metadata
            
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
            if not table:
                return metadata
                
            rows = table.find_all('tr')
            metadata['row_count'] = len(rows)
            
            if rows:
                # 检测表头
                first_row = rows[0]
                headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
                metadata['headers'] = headers
                metadata['col_count'] = len(headers)
                
                # 检测复杂结构（rowspan/colspan）
                has_span = any(cell.get('rowspan') or cell.get('colspan') 
                              for row in rows for cell in row.find_all(['th', 'td']))
                if has_span:
                    metadata['structure_type'] = 'hierarchical'
                
                # 提取所有cell值用于后续匹配
                for row in rows[1:]:  # 跳过表头
                    cells = [td.get_text().strip() for td in row.find_all(['th', 'td'])]
                    metadata['cell_values'].extend(cells)
                    
        except Exception as e:
            logger.debug(f"表格元数据提取失败: {e}")
            
        return metadata
    
    def _postprocess_triples_enhanced(self, triples: List[Tuple[str, str, str]], original_table: str, 
                                     table_text: str, metadata: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        增强的三元组后处理：基于表格元数据进行智能验证和过滤
        
        Args:
            triples: 原始三元组列表
            original_table: 原始表格HTML
            table_text: 已解析的表格文本
            metadata: 表格元数据
            
        Returns:
            处理后的三元组列表
        """
        
        processed_triples = []
        seen_triples = set()
        
        # 构建表格语义词汇表，用于更智能的匹配
        semantic_vocabulary = self._build_semantic_vocabulary(metadata)
        
        for subject, predicate, obj in triples:
            # 创建三元组的唯一标识
            triple_key = (subject.lower(), predicate.lower(), obj.lower())
            
            # 去重
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            
            # 增强的验证策略
            validation_score = self._calculate_triple_relevance(
                subject, predicate, obj, table_text, metadata, semantic_vocabulary
            )
            
            # 使用动态阈值（根据表格复杂度调整）
            threshold = self._get_dynamic_threshold(metadata)
            
            if validation_score >= threshold:
                processed_triples.append((subject, predicate, obj))
            else:
                logger.debug(f"过滤掉低相关性三元组 (分数: {validation_score:.2f}): ({subject}, {predicate}, {obj})")
        
        logger.info(f"增强后处理完成：保留 {len(processed_triples)}/{len(triples)} 个三元组")
        return processed_triples
    
    def _build_semantic_vocabulary(self, metadata: Dict[str, Any]) -> Set[str]:
        """
        构建表格的语义词汇表
        
        Args:
            metadata: 表格元数据
            
        Returns:
            语义词汇集合
        """
        vocabulary = set()
        
        # 添加表头
        for header in metadata.get('headers', []):
            vocabulary.add(header.lower())
            # 分词添加
            vocabulary.update(header.lower().split())
            
        # 添加cell值
        for cell_value in metadata.get('cell_values', []):
            if len(cell_value) > 1:  # 过滤单字符
                vocabulary.add(cell_value.lower())
                # 分词添加（对于复合词）
                if len(cell_value.split()) > 1:
                    vocabulary.update(cell_value.lower().split())
                    
        return vocabulary
    
    def _calculate_triple_relevance(self, subject: str, predicate: str, obj: str, 
                                  table_text: str, metadata: Dict[str, Any], 
                                  vocabulary: Set[str]) -> float:
        """
        计算三元组与表格的相关性分数
        
        Args:
            subject: 主体
            predicate: 谓词
            obj: 客体
            table_text: 表格文本
            metadata: 表格元数据
            vocabulary: 语义词汇表
            
        Returns:
            相关性分数 (0-1)
        """
        score = 0.0
        
        # 1. 直接匹配 (权重: 0.4)
        subject_match = subject.lower() in table_text
        obj_match = obj.lower() in table_text
        if subject_match and obj_match:
            score += 0.4
        elif subject_match or obj_match:
            score += 0.2
            
        # 2. 语义词汇匹配 (权重: 0.3)
        subject_vocab_match = any(word in vocabulary for word in subject.lower().split())
        obj_vocab_match = any(word in vocabulary for word in obj.lower().split())
        if subject_vocab_match and obj_vocab_match:
            score += 0.3
        elif subject_vocab_match or obj_vocab_match:
            score += 0.15
            
        # 3. 语义节点特殊处理 (权重: 0.2)
        is_semantic_node = '_' in subject or '_' in obj
        if is_semantic_node:
            # 语义节点应该得到更多支持
            parts_match = 0
            for entity in [subject, obj]:
                if '_' in entity:
                    parts = entity.replace('_', ' ').split()
                    matched_parts = sum(1 for part in parts if part.lower() in table_text)
                    if matched_parts > 0:
                        parts_match += matched_parts / len(parts)
            score += 0.2 * min(1.0, parts_match)
            
        # 4. 谓词相关性 (权重: 0.1)
        # 检查谓词是否与表格主题相关
        predicate_words = predicate.lower().split()
        predicate_relevance = any(word in vocabulary for word in predicate_words)
        if predicate_relevance:
            score += 0.1
            
        return min(1.0, score)
    
    def _get_dynamic_threshold(self, metadata: Dict[str, Any]) -> float:
        """
        根据表格复杂度获取动态过滤阈值
        
        Args:
            metadata: 表格元数据
            
        Returns:
            过滤阈值
        """
        base_threshold = 0.3
        
        # 根据表格复杂度调整
        if metadata.get('structure_type') == 'hierarchical':
            # 复杂表格降低阈值，保留更多可能有用的三元组
            return base_threshold - 0.1
        elif metadata.get('row_count', 0) > 10:
            # 大表格降低阈值
            return base_threshold - 0.05
        else:
            return base_threshold
    
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
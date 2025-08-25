import os
from typing import List
import json
import argparse
import logging
import requests
from typing import Optional, Dict
import csv
import logging
import re

from src.hipporag import HippoRAG

os.environ['OPENAI_API_KEY'] = 'sk-6a44d15e56dd4007945ccc41b97b499c'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_questions(file_path: str) -> List[str]:
    """从文件中加载测试问题"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 提取问题（包含数字后跟句号的行）
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s+', line):
                # 移除数字前缀
                line = re.sub(r'^\d+\.\s+', '', line)
            if line:
                questions.append(line)
                # 清洗内容
                # cleaned_line = clean_content(line)
                # if cleaned_line:
                #     questions.append(cleaned_line)
    
    return questions

def generate_csv_data(query_solutions: List[Dict], qa_top_k: int) -> List[Dict]:
    """生成CSV数据"""
    csv_data = []
    
    for i, query_solution in enumerate(query_solutions, 1):
        query_id = f"query_{i}"
        question = query_solution.question
        answer = query_solution.answer
        docs = query_solution.docs
        
        # 过滤掉空资源
        valid_resources = []
        for resource in docs:
            if resource.strip():
                valid_resources.append(resource)
        # 只保留前qa_top_k个检索文档
        valid_resources = valid_resources[:qa_top_k]
        
        if not valid_resources:
            # 如果没有有效的检索资源，创建一个包含最小段落的单行
            csv_data.append({
                'query_id': query_id,
                'query': question,
                'query_run': 1,
                'passage_id': '[1]',
                'passage': 'No relevant passages found.',
                'generated_answer': answer
            })
        else:
            # 为每个有效的检索资源创建多行
            for j, resource in enumerate(valid_resources, 1):
                
                csv_data.append({
                    'query_id': query_id,
                    'query': question,
                    'query_run': 1,
                    'passage_id': f'[{j}]',
                    'passage': resource,
                    'generated_answer': answer if j == 1 else ''
                })
    
    return csv_data

def save_to_csv(csv_data: List[Dict], output_file: str):
    """将数据保存到CSV文件"""
    # 确保字段顺序正确
    fieldnames = ['query_id', 'query', 'query_run', 'passage_id', 'passage', 'generated_answer']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="HippoRAG with configurable table processing modes")
    
    # 基础参数
    parser.add_argument('--save_dir', type=str, default='outputs/openai', 
                       help='Directory to save HippoRAG objects')
    parser.add_argument('--llm_model_name', type=str, default='qwen-plus',
                       help='LLM model name')
    parser.add_argument('--llm_base_url', type=str, 
                       default="https://dashscope.aliyuncs.com/compatible-mode/v1/",
                       help='LLM base URL')
    parser.add_argument('--embedding_model_name', type=str, default='/data/models/bge-m3',
                       help='Embedding model name')
    
    # 表格处理相关参数
    parser.add_argument('--table_processing_mode', type=str, 
                       choices=['triple_extraction', 'text_conversion'],
                       default='triple_extraction',
                       help='Mode for processing tables: triple_extraction or text_conversion')
    parser.add_argument('--text_conversion_chunk_size', type=int, default=500,
                       help='Chunk size for text conversion mode')
    parser.add_argument('--text_conversion_overlap', type=int, default=50,
                       help='Overlap size for text chunks')
    parser.add_argument('--text_conversion_detail_level', type=str,
                       choices=['basic', 'detailed'], default='detailed',
                       help='Detail level for table text conversion')
    
    # 性能参数
    parser.add_argument('--embedding_batch_size', type=int, default=1,
                       help='Embedding batch size')
    parser.add_argument('--retrieval_top_k', type=int, default=200,
                       help='Top K for retrieval')
    parser.add_argument('--qa_top_k', type=int, default=5,
                       help='Top K for QA')
    parser.add_argument('--table_extraction_max_workers', type=int, default=32,
                       help='Max workers for table processing')
    
    # 数据参数
    parser.add_argument('--data_file', type=str, default="data/file_segments.json",
                       help='Path to data file containing document segments')
    parser.add_argument('--test_questions', type=str, default="test_questions/questions_test.txt",
                       help='Path to test questions file')
    parser.add_argument('--output_file', type=str, default="zycx/hipporag_results.csv",
                       help='Output CSV file path')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 输出当前配置信息
    logger.info("HippoRAG 配置信息:")
    logger.info(f"  表格处理模式: {args.table_processing_mode}")
    if args.table_processing_mode == 'text_conversion':
        logger.info(f"  文本转换chunk大小: {args.text_conversion_chunk_size}")
        logger.info(f"  文本转换重叠大小: {args.text_conversion_overlap}")
        logger.info(f"  文本转换详细程度: {args.text_conversion_detail_level}")
    logger.info(f"  LLM模型: {args.llm_model_name}")
    logger.info(f"  嵌入模型: {args.embedding_model_name}")
    
    # Prepare datasets and evaluation：通过 urls 列表拉取所有分段
    docs: List[str] = []
    content_types: List[str] = []
    logger.info(f"正在从 {args.data_file} 加载分段...")
    
    try:
        with open(args.data_file, "r") as f:
            file_segments = json.load(f)["segments"]
        for file_segment in file_segments:
            docs.append(file_segment['content'])
            content_types.append(file_segment['content_type'])
        logger.info(f"成功加载 {len(docs)} 个文档段落")
        logger.info(f"其中文本段落: {content_types.count('text')} 个, 表格段落: {content_types.count('table')} 个")
    except FileNotFoundError:
        logger.error(f"数据文件 {args.data_file} 不存在")
        return
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return

    # 创建配置
    from src.hipporag.utils.config_utils import BaseConfig
    config = BaseConfig()
    config.embedding_model_name = args.embedding_model_name
    config.embedding_batch_size = args.embedding_batch_size
    config.embedding_max_seq_len = 4096
    config.retrieval_top_k = args.retrieval_top_k
    config.qa_top_k = args.qa_top_k
    config.table_extraction_max_workers = args.table_extraction_max_workers
    
    # 设置表格处理模式
    config.table_processing_mode = args.table_processing_mode
    config.text_conversion_chunk_size = args.text_conversion_chunk_size
    config.text_conversion_overlap = args.text_conversion_overlap
    config.text_conversion_detail_level = args.text_conversion_detail_level

    # Startup a HippoRAG instance
    logger.info("初始化 HippoRAG 实例...")
    hipporag = HippoRAG(save_dir=args.save_dir,
                        llm_model_name=args.llm_model_name,
                        llm_base_url=args.llm_base_url,
                        embedding_model_name=args.embedding_model_name,
                        global_config=config)

    # Run indexing with table support
    logger.info("开始文档索引...")
    if hasattr(hipporag, 'index_with_tables'):
        # 使用新的支持表格的索引方法
        hipporag.index_with_tables(docs=docs, content_types=content_types)
    else:
        # 降级到传统索引方法，仅处理文本
        text_docs = [doc for doc, ctype in zip(docs, content_types) if ctype == 'text']
        if text_docs:
            logger.info(f"当前版本不支持表格处理，仅索引 {len(text_docs)} 个文本文档")
            hipporag.index(docs=text_docs)
        else:
            logger.warning("没有发现文本文档，且当前HippoRAG版本不支持表格处理")
            return

    # 加载测试问题并执行QA
    logger.info(f"从 {args.test_questions} 加载测试问题...")
    try:
        questions = load_test_questions(args.test_questions)
        logger.info(f"加载了 {len(questions)} 个测试问题")
    except FileNotFoundError:
        logger.error(f"测试问题文件 {args.test_questions} 不存在")
        return
    except Exception as e:
        logger.error(f"加载测试问题失败: {e}")
        return

    logger.info("开始执行问答...")
    query_solutions, responses, metadata = hipporag.rag_qa(queries=questions)
    
    # 生成并保存结果
    logger.info("生成和保存结果...")
    csv_data = generate_csv_data(query_solutions, config.qa_top_k)
    save_to_csv(csv_data, args.output_file)
    logger.info(f"结果已保存到 {args.output_file}")
    
    # 打印统计信息
    logger.info("处理完成! 统计信息:")
    logger.info(f"  处理问题数: {len(questions)}")
    logger.info(f"  生成答案数: {len(query_solutions)}")
    logger.info(f"  输出记录数: {len(csv_data)}")

if __name__ == "__main__":
    main()

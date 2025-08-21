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

test_questions = "test_questions/questions_test.txt"
output_file = "zycx/hipporag_results.csv"

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

def main():
    # result_volume_list = get_result_volume_list()
    # for result_volume in result_volume_list:
    #     if result_volume['name'] == result_volume_name:
    #         result_volume_id = result_volume['id']
    #         break
    # sub_volume_list = get_sub_volume_list(result_volume_id)
    # for sub_volume in sub_volume_list:
    #     if sub_volume['name'] == sub_volume_name:
    #         sub_volume_id = sub_volume['id']
    #         break
    # file_list = get_volume_files(sub_volume_id)
    # for file in file_list:
    #     file_id = file['id']
    #     segments = get_file_segments(sub_volume_id, file_id)
    #     print(segments)
    # Prepare datasets and evaluation：通过 urls 列表拉取所有分段
    docs: List[str] = []
    content_types: List[str] = []
    print(f"正在加载分段。。。")
    with open("data/file_segments.json", "r") as f:
        file_segments = json.load(f)["segments"]
    for file_segment in file_segments:
        docs.append(file_segment['content'])
        content_types.append(file_segment['content_type'])
    print("len(docs):", len(docs))
    # docs = [
    #     "Oliver Badman is a politician.",
    #     "George Rankin is a politician.",
    #     "Thomas Marwick is a politician.",
    #     "Cinderella attended the royal ball.",
    #     "The prince used the lost glass slipper to search the kingdom.",
    #     "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    #     "Erik Hort's birthplace is Montebello.",
    #     "Marina is bom in Minsk.",
    #     "Montebello is a part of Rockland County."
    # ]

    save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    
    llm_model_name = 'qwen-plus'
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    embedding_model_name = '/data/models/bge-m3'
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 创建配置以优化内存使用
    from src.hipporag.utils.config_utils import BaseConfig
    config = BaseConfig()
    config.embedding_model_name = embedding_model_name
    config.embedding_batch_size = 1  # 大幅减小批处理大小
    config.embedding_max_seq_len = 4096  # 减小最大序列长度
    # config.embedding_model_dtype = "float16"  # 使用半精度浮点数节省内存
    config.retrieval_top_k = 200
    config.qa_top_k = 5
    config.table_extraction_max_workers = 32

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=llm_base_url,
                        embedding_model_name=embedding_model_name,
                        global_config=config)

    # Run indexing
    # hipporag.index(docs=docs)
    # Run indexing with table support
    if hasattr(hipporag, 'index_with_tables'):
        # 使用新的支持表格的索引方法
        hipporag.index_with_tables(docs=docs, content_types=content_types)
    else:
        # 降级到传统索引方法，仅处理文本
        text_docs = [doc for doc, ctype in zip(docs, content_types) if ctype == 'text']
        if text_docs:
            hipporag.index(docs=text_docs)
        else:
            logger.warning("没有发现文本文档，且当前HippoRAG版本不支持表格处理")

    # Separate Retrieval & QA
    # queries = [
    #     "What is George Rankin's occupation?",
    #     "How did Cinderella reach her happy ending?",
    #     "What county is Erik Hort's birthplace a part of?"
    # ]
    questions = load_test_questions(test_questions)

    query_solutions, responses, metadata = hipporag.rag_qa(queries=questions)
    
    # for (query, solution, response, meta) in zip(questions, query_solutions, responses, metadata):
    #     print(f"query: {query}")
    #     print(f"solution: {solution[0]}")
    #     print(f"response: {response}")
    #     print(f"meta: {meta}")
    #     print("-"*100)
    csv_data = generate_csv_data(query_solutions, config.qa_top_k)
    save_to_csv(csv_data, output_file)

if __name__ == "__main__":
    main()

"""
文本分块辅助工具函数。

提供分块效果分析、可视化和性能测试等实用工具。
"""

import time
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from pathlib import Path

from .base import ChunkResult, ChunkingStrategy
from .config import ChunkingConfig

logger = logging.getLogger(__name__)


class ChunkingUtils:
    """分块工具类，提供各种实用功能"""
    
    @staticmethod
    def analyze_chunk_results(results: List[ChunkResult]) -> Dict[str, Any]:
        """
        分析分块结果的统计信息。
        
        Args:
            results: 分块结果列表
            
        Returns:
            Dict[str, Any]: 统计分析结果
        """
        if not results:
            return {'error': '没有分块结果可分析'}
        
        # 收集所有块的大小
        all_chunk_sizes = []
        all_processing_times = []
        strategy_counts = {}
        
        for result in results:
            all_chunk_sizes.extend(result.get_chunk_sizes())
            all_processing_times.append(result.processing_time)
            
            strategy = result.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 计算统计指标
        analysis = {
            'total_results': len(results),
            'total_chunks': sum(len(r.chunks) for r in results),
            'chunk_size_stats': {
                'mean': statistics.mean(all_chunk_sizes) if all_chunk_sizes else 0,
                'median': statistics.median(all_chunk_sizes) if all_chunk_sizes else 0,
                'mode': statistics.mode(all_chunk_sizes) if len(all_chunk_sizes) > 0 else 0,
                'std_dev': statistics.stdev(all_chunk_sizes) if len(all_chunk_sizes) > 1 else 0,
                'min': min(all_chunk_sizes) if all_chunk_sizes else 0,
                'max': max(all_chunk_sizes) if all_chunk_sizes else 0,
                'percentiles': {
                    '25th': ChunkingUtils._percentile(all_chunk_sizes, 25),
                    '50th': ChunkingUtils._percentile(all_chunk_sizes, 50),
                    '75th': ChunkingUtils._percentile(all_chunk_sizes, 75),
                    '90th': ChunkingUtils._percentile(all_chunk_sizes, 90),
                    '95th': ChunkingUtils._percentile(all_chunk_sizes, 95)
                } if all_chunk_sizes else {}
            },
            'processing_time_stats': {
                'total': sum(all_processing_times),
                'mean': statistics.mean(all_processing_times) if all_processing_times else 0,
                'median': statistics.median(all_processing_times) if all_processing_times else 0,
                'min': min(all_processing_times) if all_processing_times else 0,
                'max': max(all_processing_times) if all_processing_times else 0
            },
            'strategy_distribution': strategy_counts,
            'quality_metrics': ChunkingUtils._calculate_quality_metrics(results)
        }
        
        return analysis
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    @staticmethod
    def _calculate_quality_metrics(results: List[ChunkResult]) -> Dict[str, float]:
        """计算分块质量指标"""
        if not results:
            return {}
        
        # 计算块大小一致性 (变异系数)
        all_sizes = []
        for result in results:
            all_sizes.extend(result.get_chunk_sizes())
        
        size_consistency = 0.0
        if all_sizes and len(all_sizes) > 1:
            mean_size = statistics.mean(all_sizes)
            std_size = statistics.stdev(all_sizes)
            size_consistency = 1 - (std_size / mean_size) if mean_size > 0 else 0
        
        # 计算平均处理效率 (字符/秒)
        total_chars = sum(sum(len(chunk) for chunk in result.chunks) for result in results)
        total_time = sum(result.processing_time for result in results)
        processing_efficiency = total_chars / total_time if total_time > 0 else 0
        
        return {
            'size_consistency': max(0, min(1, size_consistency)),  # 0-1之间
            'processing_efficiency': processing_efficiency,
            'average_chunks_per_document': sum(result.chunk_count for result in results) / len(results)
        }
    
    @staticmethod
    def compare_strategies(text: str, chunkers: List[Any]) -> Dict[str, Any]:
        """
        比较不同分块策略的效果。
        
        Args:
            text: 测试文本
            chunkers: 分块器列表
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison_results = {}
        
        for chunker in chunkers:
            try:
                result = chunker.chunk_text(text)
                strategy_name = result.strategy.value
                
                comparison_results[strategy_name] = {
                    'chunk_count': result.chunk_count,
                    'processing_time': result.processing_time,
                    'average_chunk_size': result.get_average_chunk_size(),
                    'chunk_sizes': result.get_chunk_sizes(),
                    'summary': result.summary()
                }
                
            except Exception as e:
                logger.error(f"分块器{type(chunker).__name__}处理失败: {str(e)}")
                comparison_results[f"{type(chunker).__name__}_error"] = {'error': str(e)}
        
        # 添加比较分析
        if len(comparison_results) > 1:
            comparison_results['analysis'] = ChunkingUtils._analyze_strategy_comparison(comparison_results)
        
        return comparison_results
    
    @staticmethod
    def _analyze_strategy_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
        """分析策略比较结果"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            return {'note': '需要至少两个有效结果进行比较'}
        
        # 找出各指标的最佳策略
        analysis = {
            'fastest': min(valid_results.items(), key=lambda x: x[1]['processing_time'])[0],
            'most_chunks': max(valid_results.items(), key=lambda x: x[1]['chunk_count'])[0],
            'largest_avg_chunk': max(valid_results.items(), key=lambda x: x[1]['average_chunk_size'])[0],
            'most_consistent': None  # 可以根据需要计算块大小一致性
        }
        
        return analysis
    
    @staticmethod
    def export_results(results: List[ChunkResult], output_path: str, format: str = 'json'):
        """
        导出分块结果。
        
        Args:
            results: 分块结果列表
            output_path: 输出路径
            format: 导出格式 ('json', 'txt', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            ChunkingUtils._export_json(results, output_path)
        elif format.lower() == 'txt':
            ChunkingUtils._export_txt(results, output_path)
        elif format.lower() == 'csv':
            ChunkingUtils._export_csv(results, output_path)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        logger.info(f"分块结果已导出到: {output_path}")
    
    @staticmethod
    def _export_json(results: List[ChunkResult], output_path: Path):
        """导出为JSON格式"""
        export_data = []
        for i, result in enumerate(results):
            export_data.append({
                'document_index': i,
                'strategy': result.strategy.value,
                'chunk_count': result.chunk_count,
                'processing_time': result.processing_time,
                'chunks': result.chunks,
                'metadata': result.metadata,
                'summary': result.summary()
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _export_txt(results: List[ChunkResult], output_path: Path):
        """导出为文本格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"=== 文档 {i+1} ===\\n")
                f.write(f"策略: {result.strategy.value}\\n")
                f.write(f"块数量: {result.chunk_count}\\n")
                f.write(f"处理时间: {result.processing_time:.2f}秒\\n")
                f.write("\\n--- 分块内容 ---\\n")
                
                for j, chunk in enumerate(result.chunks):
                    f.write(f"\\n[块 {j+1}]\\n{chunk}\\n")
                
                f.write("\\n" + "="*50 + "\\n\\n")
    
    @staticmethod
    def _export_csv(results: List[ChunkResult], output_path: Path):
        """导出为CSV格式"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['document_index', 'strategy', 'chunk_index', 'chunk_size', 'chunk_content'])
            
            for doc_idx, result in enumerate(results):
                for chunk_idx, chunk in enumerate(result.chunks):
                    writer.writerow([
                        doc_idx,
                        result.strategy.value,
                        chunk_idx,
                        len(chunk),
                        chunk
                    ])
    
    @staticmethod
    def validate_chunks(chunks: List[str], original_text: str) -> Dict[str, Any]:
        """
        验证分块结果的完整性和正确性。
        
        Args:
            chunks: 分块结果
            original_text: 原始文本
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # 检查空块
        empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.strip()]
        if empty_chunks:
            validation_result['issues'].append(f"发现{len(empty_chunks)}个空块: {empty_chunks}")
        
        # 检查内容完整性（简单版本）
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        original_length = len(original_text)
        
        # 允许一定的长度差异（考虑到可能的格式化差异）
        length_diff_ratio = abs(total_chunk_length - original_length) / original_length if original_length > 0 else 0
        
        if length_diff_ratio > 0.1:  # 超过10%的差异
            validation_result['issues'].append(f"内容长度差异过大: {length_diff_ratio:.2%}")
        
        # 检查重复内容（简单版本）
        chunk_hashes = set()
        duplicate_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hash(chunk.strip())
            if chunk_hash in chunk_hashes:
                duplicate_chunks.append(i)
            else:
                chunk_hashes.add(chunk_hash)
        
        if duplicate_chunks:
            validation_result['issues'].append(f"发现重复块: {duplicate_chunks}")
        
        # 设置验证状态
        validation_result['is_valid'] = len(validation_result['issues']) == 0
        
        # 添加指标
        validation_result['metrics'] = {
            'original_length': original_length,
            'total_chunk_length': total_chunk_length,
            'length_diff_ratio': length_diff_ratio,
            'empty_chunk_count': len(empty_chunks),
            'duplicate_chunk_count': len(duplicate_chunks),
            'average_chunk_size': total_chunk_length / len(chunks) if chunks else 0
        }
        
        return validation_result
    
    @staticmethod
    def benchmark_chunker(chunker, test_texts: List[str], iterations: int = 3) -> Dict[str, Any]:
        """
        对分块器进行性能基准测试。
        
        Args:
            chunker: 分块器实例
            test_texts: 测试文本列表
            iterations: 测试轮数
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        logger.info(f"开始基准测试，{len(test_texts)}个文本，{iterations}轮测试")
        
        all_results = []
        
        for iteration in range(iterations):
            iteration_start = time.time()
            iteration_results = []
            
            for text in test_texts:
                result = chunker.chunk_text(text)
                iteration_results.append(result)
            
            iteration_time = time.time() - iteration_start
            all_results.append({
                'iteration': iteration + 1,
                'results': iteration_results,
                'total_time': iteration_time
            })
        
        # 分析基准测试结果
        analysis = ChunkingUtils._analyze_benchmark_results(all_results)
        
        return {
            'test_config': {
                'text_count': len(test_texts),
                'iterations': iterations,
                'chunker_type': type(chunker).__name__
            },
            'detailed_results': all_results,
            'analysis': analysis
        }
    
    @staticmethod
    def _analyze_benchmark_results(all_results: List[Dict]) -> Dict[str, Any]:
        """分析基准测试结果"""
        iteration_times = [r['total_time'] for r in all_results]
        
        # 收集所有处理时间
        all_processing_times = []
        all_chunk_counts = []
        
        for iteration_result in all_results:
            for result in iteration_result['results']:
                all_processing_times.append(result.processing_time)
                all_chunk_counts.append(result.chunk_count)
        
        return {
            'iteration_times': {
                'mean': statistics.mean(iteration_times),
                'std_dev': statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0,
                'min': min(iteration_times),
                'max': max(iteration_times)
            },
            'processing_times': {
                'mean': statistics.mean(all_processing_times) if all_processing_times else 0,
                'std_dev': statistics.stdev(all_processing_times) if len(all_processing_times) > 1 else 0
            },
            'chunk_counts': {
                'mean': statistics.mean(all_chunk_counts) if all_chunk_counts else 0,
                'std_dev': statistics.stdev(all_chunk_counts) if len(all_chunk_counts) > 1 else 0
            },
            'stability': {
                'time_coefficient_of_variation': statistics.stdev(iteration_times) / statistics.mean(iteration_times) if iteration_times and statistics.mean(iteration_times) > 0 else 0
            }
        }
    
    @staticmethod
    def generate_test_report(analysis: Dict[str, Any], output_path: str = None) -> str:
        """
        生成分块测试报告。
        
        Args:
            analysis: 分析结果
            output_path: 输出路径（可选）
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "# 文本分块测试报告",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 总体统计",
            f"- 总文档数: {analysis.get('total_results', 'N/A')}",
            f"- 总块数: {analysis.get('total_chunks', 'N/A')}",
            ""
        ]
        
        # 块大小统计
        if 'chunk_size_stats' in analysis:
            stats = analysis['chunk_size_stats']
            report_lines.extend([
                "## 块大小统计",
                f"- 平均大小: {stats.get('mean', 0):.1f} 字符",
                f"- 中位数: {stats.get('median', 0):.1f} 字符",
                f"- 标准差: {stats.get('std_dev', 0):.1f}",
                f"- 范围: {stats.get('min', 0)} - {stats.get('max', 0)} 字符",
                ""
            ])
        
        # 性能统计
        if 'processing_time_stats' in analysis:
            time_stats = analysis['processing_time_stats']
            report_lines.extend([
                "## 性能统计",
                f"- 总处理时间: {time_stats.get('total', 0):.2f} 秒",
                f"- 平均处理时间: {time_stats.get('mean', 0):.3f} 秒/文档",
                f"- 处理速度范围: {time_stats.get('min', 0):.3f} - {time_stats.get('max', 0):.3f} 秒",
                ""
            ])
        
        # 质量指标
        if 'quality_metrics' in analysis:
            quality = analysis['quality_metrics']
            report_lines.extend([
                "## 质量指标",
                f"- 块大小一致性: {quality.get('size_consistency', 0):.3f}",
                f"- 处理效率: {quality.get('processing_efficiency', 0):.1f} 字符/秒",
                f"- 平均块数/文档: {quality.get('average_chunks_per_document', 0):.1f}",
                ""
            ])
        
        report_content = "\\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"测试报告已保存到: {output_path}")
        
        return report_content
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig


logger = get_logger(__name__)


class RetrievalRecall(BaseMetric):
    """
    检索召回率 (Retrieval Recall) 评估器。

    该类用于评估检索系统的性能，主要指标是 Recall@K。
    Recall@K 衡量的是在前 K 个检索结果中，包含了多少比例的相关文档（Gold Docs）。
    """

    metric_name: str = "retrieval_recall"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        """
        初始化 RetrievalRecall 评估器。

        参数:
            global_config (Optional[BaseConfig]): 全局配置对象。
        """
        super().__init__(global_config)

    def calculate_metric_scores(
        self,
        gold_docs: List[List[str]],
        retrieved_docs: List[List[str]],
        k_list: List[int] = [1, 5, 10, 20],
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        计算每个样本的 Recall@k，并汇总所有查询的平均结果。

        参数:
            gold_docs (List[List[str]]): 包含每个查询的标准答案（相关文档）的列表。
            retrieved_docs (List[List[str]]): 包含每个查询的检索文档的列表。
            k_list (List[int]): 需要计算 Recall@k 的 k 值列表。默认为 [1, 5, 10, 20]。

        返回:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - 一个字典，包含所有样本平均后的 Recall@k 分数（例如 {"Recall@1": 0.5, ...}）。
                - 一个列表，包含每个样本单独的 Recall@k 分数字典。
        """
        # 确保 k_list 是排序过的且无重复
        k_list = sorted(set(k_list))

        example_eval_results = []
        # 初始化汇总结果字典
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

        # 遍历每个查询的 标准文档 和 检索文档
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            # 警告：如果检索到的文档数量少于最大的 k 值，可能会影响评估结果
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(
                    f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})"
                )

            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

            # 对每个 k 值计算 Recall@k
            for k in k_list:
                # 获取前 k 个检索到的文档
                top_k_docs = example_retrieved_docs[:k]
                # 计算检索文档与标准文档的交集（即检索到的相关文档）
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # 计算召回率：检索到的相关文档数 / 总相关文档数
                if example_gold_docs:  # 避免除以零
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(
                        set(example_gold_docs)
                    )
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0

            # 添加当前样本的评估结果
            example_eval_results.append(example_eval_result)

            # 累加到汇总结果中
            for k in k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # 计算所有样本的平均值
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        # 将汇总结果保留 4 位小数
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results

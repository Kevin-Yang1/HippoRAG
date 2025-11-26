from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer

logger = get_logger(__name__)


# Reference: MRQA official eval
# 参考：MRQA 官方评估脚本
class QAExactMatch(BaseMetric):
    """
    QA 精确匹配 (Exact Match, EM) 评估指标类。
    用于衡量预测答案是否与标准答案完全一致（经过标准化处理后）。
    """

    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        aggregation_fn: Callable = np.max,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        计算精确匹配 (Exact Match, EM) 分数。

        参数:
            gold_answers (List[List[str]]): 标准答案列表的列表。每个问题可能有多个可接受的标准答案。
            predicted_answers (List[str]): 预测答案列表。
            aggregation_fn (Callable): 聚合函数，用于在多个标准答案的得分中取值（默认为 np.max，即取最高分）。

        返回:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - 一个包含平均 EM 分数的字典。
                - 一个包含每个样本 EM 分数的字典列表。
        """
        assert len(gold_answers) == len(
            predicted_answers
        ), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        # 遍历每个问题的标准答案集和预测答案
        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # 对每个标准答案，检查是否与预测答案（标准化后）完全一致
            # normalize_answer 通常会去除标点、文章冠词并统一大小写
            em_scores = [
                1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0
                for gold in gold_list
            ]

            # 使用聚合函数（如 max）计算该问题的最终得分
            # 只要预测答案匹配了任意一个标准答案，通常就得 1 分
            aggregated_em = aggregation_fn(em_scores)

            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        # 计算所有问题的平均 EM 分数
        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results


class QAF1Score(BaseMetric):
    """
    QA F1 分数 (F1 Score) 评估指标类。
    用于衡量预测答案与标准答案在词袋级别（Token-level）的重叠程度。
    """

    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        aggregation_fn: Callable = np.max,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        计算 F1 分数。

        参数:
            gold_answers (List[List[str]]): 标准答案列表的列表。
            predicted_answers (List[str]): 预测答案列表。
            aggregation_fn (Callable): 聚合函数（默认为 np.max）。

        返回:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - 一个包含平均 F1 分数的字典。
                - 一个包含每个样本 F1 分数的字典列表。
        """
        assert len(gold_answers) == len(
            predicted_answers
        ), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            """
            计算单个标准答案与预测答案之间的 F1 分数。
            """
            # 将答案标准化并按空格分词
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()

            # 计算共同出现的词（Token）数量
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            # 如果没有共同词，F1 为 0
            if num_same == 0:
                return 0.0

            # 计算精确率 (Precision) 和召回率 (Recall)
            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)

            # 计算 F1 分数：2 * (P * R) / (P + R)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        # 遍历每个问题
        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # 计算预测答案与每个标准答案的 F1 分数
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]

            # 取最高分作为该问题的最终得分
            aggregated_f1 = aggregation_fn(f1_scores)

            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        # 计算平均 F1 分数
        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results

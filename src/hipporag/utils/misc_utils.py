from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    """
    NER（命名实体识别）的原始输出数据类。
    
    Attributes:
        chunk_id (str): 文本块的唯一标识符。
        response (str): 模型的原始响应文本。
        unique_entities (List[str]): 从文本块中提取的唯一实体列表。
        metadata (Dict[str, Any]): 其他元数据信息。
    """
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    """
    三元组抽取的原始输出数据类。
    
    Attributes:
        chunk_id (str): 文本块的唯一标识符。
        response (str): 模型的原始响应文本。
        triples (List[List[str]]): 提取的三元组列表，每个三元组通常为 [subject, predicate, object]。
        metadata (Dict[str, Any]): 其他元数据信息。
    """
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    """
    链接步骤的输出数据类。
    
    Attributes:
        score (np.ndarray): 链接得分数组。
        type (Literal['node', 'dpr']): 链接类型，'node' 表示节点链接，'dpr' 表示稠密段落检索。
    """
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    """
    查询解决方案数据类，包含查询、检索结果和答案。
    
    Attributes:
        question (str): 用户的问题。
        docs (List[str]): 检索到的文档内容列表。
        doc_scores (np.ndarray): 检索到的文档对应的得分。
        answer (str): 生成的答案。
        gold_answers (List[str]): 标准答案列表（用于评估）。
        gold_docs (Optional[List[str]]): 标准文档列表（用于评估）。
    """
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        """将对象转换为字典格式，便于序列化或打印。"""
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    """
    对文本进行预处理：转小写，移除非字母数字字符。
    
    Args:
        text (Union[str, List]): 输入文本或文本列表。
        
    Returns:
        Union[str, List]: 处理后的文本或文本列表。
    """
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
    """
    重新格式化 OpenIE 结果，将其转换为 NerRawOutput 和 TripleRawOutput 对象字典。
    
    Args:
        corpus_openie_results (List[dict]): 原始 OpenIE 结果列表。
        
    Returns:
        Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]: 
            包含 NER 输出字典和三元组输出字典的元组，键为 chunk_id。
    """

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> Tuple[List[str], List[List[str]]]:
    """
    从三元组列表中提取所有唯一的实体节点。
    
    Args:
        chunk_triples (List[List[Triple]]): 每个块的三元组列表。
        
    Returns:
        Tuple[List[str], List[List[str]]]: 
            - 所有唯一实体的列表（用于构建图节点）。
            - 每个块对应的实体列表。
    """
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    """
    将嵌套的三元组列表展平为唯一事实列表。
    
    Args:
        chunk_triples (List[Triple]): 嵌套的三元组列表。
        
    Returns:
        List[Triple]: 唯一的三元组列表。
    """
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    """
    对数组进行最小-最大归一化。
    
    Args:
        x (np.ndarray): 输入数组。
        
    Returns:
        np.ndarray: 归一化后的数组，值在 [0, 1] 之间。
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    计算给定内容字符串的 MD5 哈希值，并可选择添加前缀。

    Args:
        content (str): 要哈希的输入字符串。
        prefix (str, optional): 要添加到结果哈希前面的字符串。默认为空字符串。

    Returns:
        str: 由前缀和 MD5 哈希的十六进制表示组成的字符串。
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    如果 'data' 中的所有值都具有相同的长度或 data 为空字典，则返回 True，
    否则返回 False。
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    """
    将字符串转换为布尔值。
    支持 'yes', 'true', 't', 'y', '1' 为 True，
    'no', 'false', 'f', 'n', '0' 为 False。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

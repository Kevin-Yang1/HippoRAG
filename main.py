import os
from typing import List
import json

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    """
    从样本数据中提取标准答案文档（Gold Documents）。

    该函数根据不同的数据集格式（如 HotpotQA, 2WikiMultihopQA, MuSiQue 等），
    解析样本中的支撑事实或上下文，构建包含标题和内容的标准文档列表。
    这些文档用于评估检索系统的召回率（Recall）。

    参数:
        samples (List): 包含数据集样本的列表，每个样本是一个字典。
        dataset_name (str, optional): 数据集名称，用于区分特定数据集的处理逻辑（如 hotpotqa）。

    返回:
        List: 一个列表，其中每个元素是对应样本的标准文档列表（List[str]）。
              每个文档字符串通常格式为 "标题\n内容"。
    """
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:  # 适用于 hotpotqa, 2wikimultihopqa 数据集
            # 提取支撑事实的标题集合
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            # 从上下文中筛选出标题在支撑事实中的段落
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]

            if dataset_name.startswith("hotpotqa"):
                # HotpotQA: 内容是句子列表，直接拼接
                gold_doc = [
                    item[0] + "\n" + "".join(item[1])
                    for item in gold_title_and_content_list
                ]
            else:
                # 2WikiMultihopQA: 内容是句子列表，用空格拼接
                gold_doc = [
                    item[0] + "\n" + " ".join(item[1])
                    for item in gold_title_and_content_list
                ]
        elif "contexts" in sample:
            # 适用于某些特定格式的数据集，直接包含 contexts 字段
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            # 适用于 MuSiQue 等数据集，包含 paragraphs 字段
            assert (
                "paragraphs" in sample
            ), "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                # 仅保留标记为支撑段落的项
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            # 构建文档字符串，处理 text 或 paragraph_text 字段名差异
            gold_doc = [
                item["title"]
                + "\n"
                + (item["text"] if "text" in item else item["paragraph_text"])
                for item in gold_paragraphs
            ]

        # 去重并添加到结果列表
        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    """
    从样本数据中提取标准答案（Gold Answers）。

    该函数遍历样本列表，根据不同的字段名（answer, gold_ans, reference, obj 等）
    提取问题的标准答案。它还处理了答案的别名（aliases），将它们合并到一个集合中。
    这些答案用于评估问答系统的准确率（Exact Match, F1 Score）。

    参数:
        samples (List): 包含数据集样本的列表。

    返回:
        List: 一个列表，其中每个元素是对应样本的标准答案集合（Set[str]）。
    """
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        # 尝试从不同字段获取答案
        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            # 针对特定数据集（可能是知识图谱问答），合并对象、可能答案、Wiki标题和别名
            gold_ans = set(
                [sample["obj"]]
                + [sample["possible_answers"]]
                + [sample["o_wiki_title"]]
                + [sample["o_aliases"]]
            )
            gold_ans = list(gold_ans)

        assert gold_ans is not None

        # 确保答案是列表格式
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)

        # 转换为集合以去重
        gold_ans = set(gold_ans)

        # 如果存在答案别名，也加入到标准答案集合中
        if "answer_aliases" in sample:
            gold_ans.update(sample["answer_aliases"])

        gold_answers.append(gold_ans)

    return gold_answers


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="LLM base URL",
    )
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument(
        "--embedding_name",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="embedding model name",
    )
    parser.add_argument(
        "--force_index_from_scratch",
        type=str,
        default="false",
        help="If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.",
    )
    parser.add_argument(
        "--force_openie_from_scratch",
        type=str,
        default="false",
        help="If set to False, will try to first reuse openie results for the corpus if they exist.",
    )
    parser.add_argument(
        "--openie_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Save directory"
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    if save_dir == "outputs":
        save_dir = save_dir + "/" + dataset_name
    else:
        save_dir = save_dir + "_" + dataset_name

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s["question"] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert (
            len(all_queries) == len(gold_docs) == len(gold_answers)
        ), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()

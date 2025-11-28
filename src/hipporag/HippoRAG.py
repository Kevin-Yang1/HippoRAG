import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import (
    TransformersOfflineOpenIE,
)
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)


class HippoRAG:

    def __init__(
        self,
        global_config=None,
        save_dir=None,
        llm_model_name=None,
        llm_base_url=None,
        embedding_model_name=None,
        embedding_base_url=None,
        azure_endpoint=None,
        azure_embedding_endpoint=None,
    ):
        """
        初始化 HippoRAG 实例及其相关组件。

        属性:
            global_config (BaseConfig): 实例的全局配置设置。如果未提供值，则使用 BaseConfig 的实例。
            saving_dir (str): 存储特定 HippoRAG 实例的目录。如果未提供值，默认为 `outputs`。
            llm_model (BaseLLM): 基于全局配置设置用于处理的语言模型。
            openie (Union[OpenIE, VLLMOfflineOpenIE]): 开放信息抽取模块，根据全局设置配置为在线或离线模式。
            graph: 由 `initialize_graph` 方法初始化的图实例。
            embedding_model (BaseEmbeddingModel): 与当前配置关联的嵌入模型。
            chunk_embedding_store (EmbeddingStore): 处理段落嵌入的嵌入存储。
            entity_embedding_store (EmbeddingStore): 处理实体嵌入的嵌入存储。
            fact_embedding_store (EmbeddingStore): 处理事实嵌入的嵌入存储。
            prompt_template_manager (PromptTemplateManager): 用于处理提示模板和角色映射的管理器。
            openie_results_path (str): 基于全局配置中的数据集和 LLM 名称存储开放信息抽取结果的文件路径。
            rerank_filter (Optional[DSPyFilter]): 当全局配置中指定了重排序文件路径时，负责重排序信息的过滤器。
            ready_to_retrieve (bool): 指示系统是否准备好进行检索操作的标志。

        参数:
            global_config: 全局配置对象。默认为 None，此时会初始化一个新的 BaseConfig 对象。
            save_dir: 存储工作文件的目录。
            llm_model_name: LLM 模型名称，可以直接插入，也可以通过配置文件插入。
            embedding_model_name: 嵌入模型名称，可以直接插入，也可以通过配置文件插入。
            llm_base_url: 已部署 LLM 模型的 URL，可以直接插入，也可以通过配置文件插入。
            embedding_base_url: 嵌入模型的 URL。
            azure_endpoint: Azure 端点。
            azure_embedding_endpoint: Azure 嵌入端点。
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # 如果指定了参数，则覆盖配置
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join(
            [f"{k} = {v}" for k, v in asdict(self.global_config).items()]
        )
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        # 在每个指定的保存目录下创建特定于 LLM 和嵌入模型的工作目录
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(
            self.global_config.save_dir, f"{llm_label}_{embedding_label}"
        )

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        # 初始化 LLM 模型
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        # 根据配置初始化 OpenIE 模块（在线、离线 VLLM 或离线 Transformers）
        if self.global_config.openie_mode == "online":
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == "offline":
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode == "Transformers-offline":
            self.openie = TransformersOfflineOpenIE(self.global_config)

        # 初始化图结构
        self.graph = self.initialize_graph()

        # 初始化嵌入模型（如果是离线 OpenIE 模式则跳过）
        if self.global_config.openie_mode == "offline":
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name
            )(
                global_config=self.global_config,
                embedding_model_name=self.global_config.embedding_model_name,
            )

        # 初始化向量存储（段落、实体、事实）
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size,
            "chunk",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "entity_embeddings"),
            self.global_config.embedding_batch_size,
            "entity",
        )
        self.fact_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "fact_embeddings"),
            self.global_config.embedding_batch_size,
            "fact",
        )

        # 初始化提示模板管理器
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        # 设置 OpenIE 结果保存路径
        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json',
        )

        # 初始化重排序过滤器
        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(self.working_dir, f"graph.pickle")

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self, docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunks.keys()
        )
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(
                new_openie_rows
            )
            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                new_triple_results_dict,
            )

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info(
            "Done with OpenIE, run online indexing for future retrieval."
        )  # 主动报错并停止运行

    def index(self, docs: List[str]):
        """
        基于 HippoRAG 2 框架对给定文档进行索引。
        该过程会基于文档生成 OpenIE 知识图谱，并分别对段落、实体和事实进行编码，以便后续检索。

        参数:
            docs : List[str]
                待索引的文档列表。
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        # 如果是离线模式，先执行预处理（抽取并保存），然后程序会终止。
        # 下次运行时（非离线模式或已有缓存），将跳过此步骤直接加载结果。
        if self.global_config.openie_mode == "offline":
            self.pre_openie(docs)

        # 1. 将文档插入段落向量存储（Chunk Embedding Store）
        # 这里会计算文档的哈希 ID，如果已存在则不会重复插入
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        # 2. 加载已有的 OpenIE 结果(all_openie_info)，并找出需要新处理的文档块(chunk_keys_to_process)
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunk_to_rows.keys()
        )  # 这里的输入为所有的哈希IDs

        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        # 3. 对新文档块执行 OpenIE（命名实体识别 + 三元组抽取）
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(
                new_openie_rows
            )
            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                new_triple_results_dict,
            )

        # 4. 如果配置了保存 OpenIE 结果，则将其持久化到磁盘
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # 5. 格式化 OpenIE 结果，准备构建图
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert (
            len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)
        ), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # 准备数据存储
        chunk_ids = list(chunk_to_rows.keys())

        # 处理三元组文本
        chunk_triples = [
            [text_processing(t) for t in triple_results_dict[chunk_id].triples]
            for chunk_id in chunk_ids
        ]
        # 提取实体节点
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        # 展平事实列表
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        # 6. 对实体进行编码并存入实体向量存储
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        # 7. 对事实（三元组）进行编码并存入事实向量存储
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        # 8. 构建图结构
        # 添加事实边（实体之间的关系）
        self.add_fact_edges(chunk_ids, chunk_triples)
        # 添加段落边（段落与实体之间的关系）
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        # 9. 如果有新文档块，则更新图并保存
        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            # 添加同义词边（相似实体之间的连接）
            self.add_synonymy_edges()

            # 扩展图（添加新节点和边）
            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        # Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        # Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [
                self.chunk_embedding_store.text_to_hash_id[chunk]
                for chunk in docs_to_delete
            ]
        )

        # Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc["idx"] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc["extracted_triples"])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        # Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [
            [text_processing(list(triple)) for triple in true_triples_to_delete]
        ]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(
            processed_true_triples_to_delete
        )

        triple_ids_to_delete = set(
            [
                self.fact_embedding_store.text_to_hash_id[str(triple)]
                for triple in processed_true_triples_to_delete
            ]
        )

        # Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [
            self.entity_embedding_store.text_to_hash_id[ent]
            for ent in entities_to_delete
        ]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        # Delete Nodes from Graph
        self.graph.delete_vertices(
            list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete)
        )
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        使用 HippoRAG 2 框架执行检索操作。

        该过程包含以下几个关键步骤：
        1. **事实检索 (Fact Retrieval)**：从查询中提取关键事实。
        2. **识别记忆 (Recognition Memory)**：利用识别记忆机制改进事实的选择。
        3. **稠密段落打分 (Dense Passage Scoring)**：对段落进行初步的稠密向量检索打分。
        4. **个性化 PageRank 重排序 (Personalized PageRank based re-ranking)**：基于图结构进行最终的重排序。

        参数:
            queries (List[str]): 需要检索文档的查询字符串列表。
            num_to_retrieve (int, optional): 每个查询需要检索的最大文档数量。
                如果未指定，默认使用全局配置中的 `retrieval_top_k` 值。
            gold_docs (List[List[str]], optional): 每个查询对应的标准答案文档列表（Gold Standard）。
                如果启用了检索性能评估（全局配置中的 `do_eval_retrieval`），则必须提供此参数。

        返回:
            List[QuerySolution] or (List[QuerySolution], Dict):
                - 如果未启用检索性能评估，仅返回一个 `QuerySolution` 对象列表，每个对象包含对应查询的检索文档及其得分。
                - 如果启用了评估，则返回一个元组，包含上述列表以及一个包含评估指标（如 Recall）的字典。

        注意:
            - 如果长查询在重排序后没有找到相关事实，系统将回退到使用稠密段落检索 (Dense Passage Retrieval) 的结果。
        """
        retrieve_start_time = time.time()  # 记录检索开始时间

        # 如果未指定检索数量，使用全局配置的默认值
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        # 如果提供了标准文档，初始化检索召回率评估器
        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(
                global_config=self.global_config
            )

        # 确保检索所需的内存对象（如向量、图索引）已准备就绪
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        # 获取查询的向量表示
        self.get_query_embeddings(queries)

        retrieval_results = []

        # 遍历每个查询进行处理
        for q_idx, query in tqdm(
            enumerate(queries), desc="Retrieving", total=len(queries)
        ):
            rerank_start = time.time()

            # 1. 获取查询与事实的相似度得分
            query_fact_scores = self.get_fact_scores(query)

            # 2. 对事实进行重排序，筛选出 Top-K 相关事实
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                query, query_fact_scores
            )
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            # 3. 根据是否有相关事实决定检索策略
            if len(top_k_facts) == 0:
                # 如果没有找到相关事实（通常发生在长查询或无匹配事实时），回退到稠密段落检索
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                # 如果找到了相关事实，使用基于图的 PPR 算法进行检索
                # 结合了查询、筛选出的事实以及段落节点权重
                sorted_doc_ids, sorted_doc_scores = (
                    self.graph_search_with_fact_entities(
                        query=query,
                        link_top_k=self.global_config.linking_top_k,
                        query_fact_scores=query_fact_scores,
                        top_k_facts=top_k_facts,
                        top_k_fact_indices=top_k_fact_indices,
                        passage_node_weight=self.global_config.passage_node_weight,
                    )
                )

            # 获取 Top-K 文档的内容
            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])[
                    "content"
                ]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]

            # 保存当前查询的检索结果
            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

        retrieve_end_time = time.time()  # 记录检索结束时间

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        # 打印各阶段耗时统计
        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(
            f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s"
        )

        # 如果提供了标准文档，执行检索评估
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[
                        retrieval_result.docs for retrieval_result in retrieval_results
                    ],
                    k_list=k_list,
                )
            )
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ) -> (
        Tuple[List[QuerySolution], List[str], List[Dict]]
        | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
    ):
        """
        使用 HippoRAG 2 框架执行增强的检索增强生成 (RAG) 问答。

        此方法可以处理字符串形式的查询或预处理过的 QuerySolution 对象。
        根据输入的不同，它仅返回答案，或者额外评估检索和回答的质量（使用 recall@k、精确匹配 EM 和 F1 分数指标）。

        参数:
            queries (List[Union[str, QuerySolution]]): 查询列表，可以是字符串或 QuerySolution 实例。
                如果是字符串，将首先执行检索步骤。
            gold_docs (Optional[List[List[str]]]): 每个查询对应的标准答案文档列表（Gold Standard）。
                如果需要执行文档级别的检索评估，则必须提供此参数。默认为 None。
            gold_answers (Optional[List[List[str]]]): 每个查询对应的标准答案列表。
                如果启用了问答 (QA) 答案评估，则必须提供此参数。默认为 None。

        返回:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: 一个元组，总是包含：
                - List[QuerySolution]: 包含每个查询的答案和元数据的 QuerySolution 对象列表。
                - List[str]: 针对提供的查询的原始响应消息列表。
                - List[Dict]: 每个查询的元数据字典列表。
                如果启用了评估（提供了 gold_answers），元组还包括：
                - Dict: 检索阶段的总体评估结果（如果适用）。
                - Dict: QA 总体评估指标（精确匹配和 F1 分数）。
        """
        # 如果提供了标准答案，初始化 QA 评估器（精确匹配和 F1 分数）
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # 检索阶段（如果必要）
        overall_retrieval_result = None

        # 检查输入是否为原始字符串查询。如果是，则先执行检索。
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                # 如果提供了标准文档，执行带评估的检索
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                # 否则执行普通检索
                queries = self.retrieve(queries=queries)

        # 执行 QA 推理
        # queries 现在是 QuerySolution 对象列表（包含检索到的文档）
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # 评估 QA 结果
        if gold_answers is not None:
            # 计算精确匹配 (Exact Match) 分数
            overall_qa_em_result, example_qa_em_results = (
                qa_em_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )
            # 计算 F1 分数
            overall_qa_f1_result, example_qa_f1_results = (
                qa_f1_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )

            # 合并评估结果并保留 4 位小数
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {
                k: round(float(v), 4) for k, v in overall_qa_results.items()
            }
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # 将标准答案和标准文档保存到结果对象中，以便后续分析
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            # 返回包含评估结果的元组
            return (
                queries_solutions,
                all_response_message,
                all_metadata,
                overall_retrieval_result,
                overall_qa_results,
            )
        else:
            # 如果不需要评估，仅返回 QA 结果
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(
                global_config=self.global_config
            )

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(
            enumerate(queries), desc="Retrieving", total=len(queries)
        ):
            logger.info("No facts found after reranking, return DPR results")
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])[
                    "content"
                ]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[
                        retrieval_result.docs for retrieval_result in retrieval_results
                    ],
                    k_list=k_list,
                )
            )
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ) -> (
        Tuple[List[QuerySolution], List[str], List[Dict]]
        | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
    ):
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = (
                qa_em_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )
            overall_qa_f1_result, example_qa_f1_results = (
                qa_f1_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {
                k: round(float(v), 4) for k, v in overall_qa_results.items()
            }
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return (
                queries_solutions,
                all_response_message,
                all_metadata,
                overall_retrieval_result,
                overall_qa_results,
            )
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(
        self, queries: List[QuerySolution]
    ) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[: self.global_config.qa_top_k]

            prompt_user = ""
            for passage in retrieved_passages:
                prompt_user += f"Wikipedia Title: {passage}\n\n"
            prompt_user += "Question: " + query_solution.question + "\nThought: "

            if self.prompt_template_manager.is_template_name_valid(
                name=f"rag_qa_{self.global_config.dataset}"
            ):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead."
                )
                prompt_dataset_name = "musique"
            all_qa_messages.append(
                self.prompt_template_manager.render(
                    name=f"rag_qa_{prompt_dataset_name}", prompt_user=prompt_user
                )
            )

        all_qa_results = [
            self.llm_model.infer(qa_messages)
            for qa_messages in tqdm(all_qa_messages, desc="QA Reading")
        ]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(
            all_metadata
        )

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(
            enumerate(queries), desc="Extraction Answers from LLM Response"
        ):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split("Answer:")[1].strip()
            except Exception as e:
                logger.warning(
                    f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!"
                )
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(
                        content=triple[2], prefix=("entity-")
                    )

                    self.node_to_node_stats[(node_key, node_2_key)] = (
                        self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                    )
                    self.node_to_node_stats[(node_2_key, node_key)] = (
                        self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                    )

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(
                        node, set()
                    ).union(set([chunk_key]))

    def add_passage_edges(
        self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]
    ):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(
            f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)})."
        )

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size,
        )

        num_synonym_triple = 0
        synonym_candidates = (
            []
        )  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(
            query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)
        ):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub("[^A-Za-z0-9]", "", entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if (
                        score < self.global_config.synonymy_edge_sim_threshold
                        or num_nns > 100
                    ):
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != "":
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = (
                            score  # Need to seriously discuss on this
                        )
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(
        self, chunk_keys: List[str]
    ) -> Tuple[List[dict], Set[str]]:
        """
        加载现有的 OpenIE 结果（如果存在），并将其与新内容合并，同时标准化索引。
        如果文件不存在，或者配置了 `force_openie_from_scratch` 标志以强制从头开始，
        则准备新的条目进行处理。

        参数:
            chunk_keys (List[str]): 代表待处理内容的段落键（哈希 ID）列表。

        返回:
            Tuple[List[dict], Set[str]]: 一个元组，包含：
                - 第一项是从文件加载的现有 OpenIE 信息列表（如果有）。
                - 第二项是仍需保存或处理的段落键集合。
        """

        # 将 openie_results 与文件中已有的内容合并（如果文件存在）
        chunk_keys_to_save = set()

        # 检查是否强制重新运行 OpenIE，以及结果文件是否存在
        if not self.global_config.force_openie_from_scratch and os.path.isfile(
            self.openie_results_path
        ):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get("docs", [])

            # 标准化 OpenIE 文件中的索引。
            # 重新计算哈希 ID 以确保一致性（防止旧文件中的 ID 生成逻辑与当前不一致）
            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info["idx"] = compute_mdhash_id(openie_info["passage"], "chunk-")
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info  # 这里冗余了，二者实际是同一个东西

            # 获取已存在的 OpenIE 结果的键集合
            existing_openie_keys = set([info["idx"] for info in all_openie_info])

            # 找出哪些输入的 chunk_key 还没有对应的 OpenIE 结果
            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            # 如果强制重跑或文件不存在，则所有输入的 chunk_keys 都需要处理
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(
        self,
        all_openie_info: List[dict],
        chunks_to_save: Dict[str, dict],
        ner_results_dict: Dict[str, NerRawOutput],
        triple_results_dict: Dict[str, TripleRawOutput],
    ) -> List[dict]:
        """
        将 OpenIE 抽取结果（NER 和三元组）与对应的段落文本及元数据合并。

        该函数将 OpenIE 的抽取结果（包括命名实体识别 NER 的实体和关系三元组）与
        原始的文本段落（通过 chunk keys 关联）整合在一起。
        合并后的数据会被追加到 `all_openie_info` 列表中，用于后续的处理或存储。

        参数:
            all_openie_info (List[dict]): 用于存储所有段落合并后的 OpenIE 结果和元数据的列表。
                新处理的结果将被追加到这个列表中。
            chunks_to_save (Dict[str, dict]): 需要处理并保存的段落字典。
                键是段落的唯一标识符 (chunk_key)，值是包含 `content` (文本内容) 等信息的字典。
            ner_results_dict (Dict[str, NerRawOutput]): 映射段落键到其对应的 NER 抽取结果的字典。
            triple_results_dict (Dict[str, TripleRawOutput]): 映射段落键到其对应的 OpenIE 三元组抽取结果的字典。

        返回:
            List[dict]: 更新后的 `all_openie_info` 列表，包含了合并了 OpenIE 结果、元数据和段落内容的字典。
        """

        # 遍历所有需要保存的段落
        for chunk_key, row in chunks_to_save.items():
            passage = row["content"]  # 获取段落原始内容
            try:
                # 构建包含完整信息的字典：索引、原文、提取的实体、提取的三元组
                chunk_openie_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_entities": ner_results_dict[
                        chunk_key
                    ].unique_entities,  # 从 NER 结果中获取去重后的实体
                    "extracted_triples": triple_results_dict[
                        chunk_key
                    ].triples,  # 从三元组结果中获取三元组列表
                }
            except Exception as e:
                # 如果在获取结果时发生错误（例如键不存在），记录错误并保存空结果，以防程序崩溃
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_entities": [],
                    "extracted_triples": [],
                }

            # 将构建好的信息追加到总列表中
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum(
            [len(e) for chunk in all_openie_info for e in chunk["extracted_entities"]]
        )
        sum_phrase_words = sum(
            [
                len(e.split())
                for chunk in all_openie_info
                for e in chunk["extracted_entities"]
            ]
        )
        num_phrases = sum(
            [len(chunk["extracted_entities"]) for chunk in all_openie_info]
        )

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                "docs": all_openie_info,
                "avg_ent_chars": avg_ent_chars,
                "avg_ent_words": avg_ent_words,
            }

            with open(self.openie_results_path, "w") as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {
            v["name"]: v for v in self.graph.vs if "name" in v.attributes()
        }

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node["name"] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(
                n=len(next(iter(new_nodes.values()))), attributes=new_nodes
            )

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(
            edge_source_node_keys, edge_target_node_keys, edge_metadata
        ):
            if (
                source_node_id in current_node_ids
                and target_node_id in current_node_ids
            ):
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(
                    f"Edge {source_node_id} -> {target_node_id} is not valid."
                )
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = (
            graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]
        )

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(
            self.fact_embedding_store.get_all_ids()
        )

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1
            for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info["num_triples_with_passage_node"] = num_triples_with_passage_node

        graph_info["num_synonymy_triples"] = (
            len(self.node_to_node_stats)
            - graph_info["num_extracted_triples"]
            - num_triples_with_passage_node
        )

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        准备快速检索所需的各种内存对象和属性。
        这包括加载嵌入数据、建立图关系映射，并确保内存中的数据结构与底层图结构的一致性。
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        # 初始化查询到嵌入的缓存字典，分为 'triple' (事实/三元组视角) 和 'passage' (段落视角)
        self.query_to_embedding: Dict = {"triple": {}, "passage": {}}

        # 从嵌入存储中加载所有实体、段落和事实的键（ID）
        self.entity_node_keys: List = list(
            self.entity_embedding_store.get_all_ids()
        )  # 短语/实体节点键列表
        self.passage_node_keys: List = list(
            self.chunk_embedding_store.get_all_ids()
        )  # 段落节点键列表
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # 检查图中的节点数量是否符合预期（实体数 + 段落数）
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()

        if expected_node_count != actual_node_count:
            logger.warning(
                f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}"
            )
            # 如果图是空的但我们有节点数据，则需要初始化图并添加节点
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # 创建从节点名称（key）到图顶点索引（index）的映射
        try:
            igraph_name_to_idx = {
                node["name"]: idx for idx, node in enumerate(self.graph.vs)
            }  # 映射: 节点键 -> 骨干图中的索引
            self.node_name_to_vertex_idx = igraph_name_to_idx

            # 检查是否所有实体和段落节点都存在于图中
            missing_entity_nodes = [
                node_key
                for node_key in self.entity_node_keys
                if node_key not in igraph_name_to_idx
            ]
            missing_passage_nodes = [
                node_key
                for node_key in self.passage_node_keys
                if node_key not in igraph_name_to_idx
            ]

            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(
                    f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes"
                )
                # 如果有节点缺失，重建图节点并保存
                self.add_new_nodes()
                self.save_igraph()
                # 更新映射
                igraph_name_to_idx = {
                    node["name"]: idx for idx, node in enumerate(self.graph.vs)
                }
                self.node_name_to_vertex_idx = igraph_name_to_idx

            # 创建实体和段落节点在图中对应的索引列表，用于后续快速访问
            self.entity_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.entity_node_keys
            ]  # 骨干图实体节点索引列表
            self.passage_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.passage_node_keys
            ]  # 骨干图段落节点索引列表
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # 如果映射失败，初始化为空列表以防崩溃
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        # 将所有实体和段落的嵌入加载到内存中的 numpy 数组，以便进行快速矩阵运算
        self.entity_embeddings = np.array(
            self.entity_embedding_store.get_embeddings(self.entity_node_keys)
        )
        self.passage_embeddings = np.array(
            self.chunk_embedding_store.get_embeddings(self.passage_node_keys)
        )

        self.fact_embeddings = np.array(
            self.fact_embedding_store.get_embeddings(self.fact_node_keys)
        )

        # 加载现有的 OpenIE 结果，用于构建三元组到文档的映射
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        # 构建处理后的三元组到文档 ID 的反向索引
        for doc in all_openie_info:
            triples = flatten_facts([doc["extracted_triples"]])
            for triple in triples:
                if len(triple) == 3:
                    # 对三元组文本进行标准化处理
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = (
                        self.proc_triples_to_docs.get(str(proc_triple), set()).union(
                            set([doc["idx"]])
                        )
                    )

        # 如果实体节点到段落 ID 的映射尚未初始化（例如首次运行或未从文件加载），则进行构建
        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(
                all_openie_info
            )

            # 检查数据长度一致性
            if not (
                len(self.passage_node_keys)
                == len(ner_results_dict)
                == len(triple_results_dict)
            ):
                logger.warning(
                    f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}"
                )

                # 如果存在缺失的键，为它们创建空的条目以保持对齐
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[],
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id, response=None, metadata={}, triples=[]
                        )

            # 准备数据存储：处理每个段落的三元组
            chunk_triples = [
                [text_processing(t) for t in triple_results_dict[chunk_id].triples]
                for chunk_id in self.passage_node_keys
            ]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            # 添加事实边，这将填充 ent_node_to_chunk_ids 和 node_to_node_stats
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        # 标记检索对象已准备就绪
        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        获取给定查询的嵌入向量，并更新内部的查询到嵌入的映射缓存。

        该方法会检查每个查询是否已经存在于 `self.query_to_embedding` 字典的 'triple'（事实检索用）
        和 'passage'（段落检索用）键下。如果某个查询在任一映射中不存在，则使用嵌入模型对其进行编码并存储。
        这确保了后续检索步骤可以直接使用缓存的向量，避免重复计算。

        参数:
            queries (List[str] | List[QuerySolution]): 查询字符串列表或 QuerySolution 对象列表。
                如果是 QuerySolution 对象，将提取其 `question` 属性作为查询文本。
        """

        all_query_strings = []
        # 遍历查询列表，筛选出尚未完全缓存嵌入的查询
        for query in queries:
            if isinstance(query, QuerySolution) and (
                query.question not in self.query_to_embedding["triple"]
                or query.question not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query.question)
            elif (
                query not in self.query_to_embedding["triple"]
                or query not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # 获取所有未缓存查询的嵌入向量

            # 1. 编码用于事实检索（query_to_fact）的向量
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(
                all_query_strings,
                instruction=get_query_instruction("query_to_fact"),
                norm=True,
            )
            # 将结果存入 'triple' 缓存
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding["triple"][query] = embedding

            # 2. 编码用于段落检索（query_to_passage）的向量
            logger.info(
                f"Encoding {len(all_query_strings)} queries for query_to_passage."
            )
            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings,
                instruction=get_query_instruction("query_to_passage"),
                norm=True,
            )
            # 将结果存入 'passage' 缓存
            for query, embedding in zip(
                all_query_strings, query_embeddings_for_passage
            ):
                self.query_to_embedding["passage"][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        检索并计算给定查询与预存储的事实嵌入向量之间的归一化相似度分数。

        该方法首先尝试获取查询的嵌入向量（如果尚未缓存则进行编码），然后计算其与所有事实嵌入向量的点积（相似度），
        最后对分数进行归一化处理。

        参数:
            query (str): 输入的查询文本，用于计算与事实嵌入的相似度分数。

        返回:
            numpy.ndarray: 查询与事实嵌入之间的归一化相似度分数数组。
                数组的形状由事实的数量决定。如果发生错误或没有事实，则返回空数组。
        """
        # 尝试从缓存中获取查询的“三元组/事实”视角的嵌入向量
        query_embedding = self.query_to_embedding["triple"].get(query, None)

        # 如果缓存中没有，则使用嵌入模型对查询进行编码
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_fact"), norm=True
            )

        # 检查是否存在任何事实嵌入
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        try:
            # 计算事实嵌入矩阵与查询嵌入向量的点积，得到相似度分数， # 注意：如果向量已归一化（L2范数为1），点积在数学上等价于余弦相似度，且计算效率更高
            # self.fact_embeddings shape: (num_facts, embedding_dim)
            # query_embedding.T shape: (embedding_dim, 1) or (embedding_dim,)
            query_fact_scores = np.dot(
                self.fact_embeddings, query_embedding.T
            )  # 结果形状: (#facts, )

            # 如果结果是二维数组（例如列向量），则压缩为一维数组
            query_fact_scores = (
                np.squeeze(query_fact_scores)
                if query_fact_scores.ndim == 2
                else query_fact_scores
            )

            # 对分数进行最小-最大归一化，使其范围在 [0, 1] 之间
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行稠密段落检索（Dense Passage Retrieval），以查找与查询相关的文档。

        该函数使用预训练的嵌入模型处理给定的查询，生成查询嵌入向量。
        然后计算查询嵌入与所有段落嵌入之间的点积作为相似度分数，并进行归一化处理。
        最后，根据相似度分数对文档进行排序，并返回排序后的文档标识符及其对应的分数。

        参数:
            query (str): 输入的查询文本，用于检索相关的段落。

        返回:
            Tuple[np.ndarray, np.ndarray]: 包含两个元素的元组：
                - 按相关性分数降序排列的文档标识符（索引）数组。
                - 对应文档的归一化相似度分数数组。
        """
        # 尝试从缓存中获取查询的“段落”视角的嵌入向量
        query_embedding = self.query_to_embedding["passage"].get(query, None)

        # 如果缓存中没有，则使用嵌入模型对查询进行编码
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_passage"), norm=True
            )

        # 计算段落嵌入矩阵与查询嵌入向量的点积，得到相似度分数
        # self.passage_embeddings shape: (num_passages, embedding_dim)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)

        # 如果结果是二维数组，则压缩为一维数组
        query_doc_scores = (
            np.squeeze(query_doc_scores)
            if query_doc_scores.ndim == 2
            else query_doc_scores
        )

        # 对分数进行最小-最大归一化
        query_doc_scores = min_max_normalize(query_doc_scores)

        # 对文档分数进行排序（降序），获取排序后的索引
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]

        # 获取对应的排序后的分数
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(
        self,
        link_top_k: int,
        all_phrase_weights: np.ndarray,
        linking_score_map: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        过滤短语权重，仅保留 `linking_score_map` 中排名前 `link_top_k` 的短语的权重。

        该函数根据 `linking_score_map` 中的分数对短语进行排序，并保留前 `link_top_k` 个短语。
        对于未被选中的短语，其在 `all_phrase_weights` 中的权重将被重置为 0.0。
        这有助于在图搜索中聚焦于最相关的实体节点。

        参数:
            link_top_k (int): 在链接分数映射中保留的排名靠前的节点数量。
            all_phrase_weights (np.ndarray): 表示短语权重的数组，通过短语 ID 索引。
            linking_score_map (Dict[str, float]): 短语内容到其链接分数的映射，通常按分数降序排列。

        返回:
            Tuple[np.ndarray, Dict[str, float]]: 一个元组，包含：
                - 过滤后的 `all_phrase_weights` 数组，其中未选中短语的权重被设为 0.0。
                - 仅包含前 `link_top_k` 个短语的过滤后的 `linking_score_map`。
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(
            sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[
                :link_top_k
            ]
        )

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [
                compute_mdhash_id(content=top_k_phrase, prefix="entity-")
                for top_k_phrase in top_k_phrases
            ]
        )

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[str],
        passage_node_weight: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        结合事实实体和稠密检索结果，在图上执行搜索（Personalized PageRank）。

        该函数利用个性化 PageRank (PPR) 和稠密检索模型，基于事实的相似度和相关性计算文档分数。
        它将识别出的相关事实（通过短语节点）的信号与段落相似度（通过段落节点）和图结构搜索相结合，以增强结果排序。

        参数:
            query (str): 输入查询字符串，用于执行相似度和相关性计算。
            link_top_k (int): 从链接分数映射中保留的顶部短语数量，用于后续处理。
            query_fact_scores (np.ndarray): 表示每个提供的事实与查询之间相似度的分数数组。
            top_k_facts (List[Tuple]): 排名靠前的事实列表，每个事实表示为 (主语, 谓语, 宾语) 的元组。
            top_k_fact_indices (List[str]): query_fact_scores 数组中对应排名靠前事实的索引或标识符。
            passage_node_weight (float): 用于缩放图中段落分数的默认权重。

        返回:
            Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
                - 第一个数组是根据分数排序的文档 ID。
                - 第二个数组是与排序后的文档 ID 对应的 PPR 分数。
        """

        # 根据前一步选出的事实分配短语权重。
        linking_score_map = {}  # 映射短语到包含该短语的事实的平均分数
        phrase_scores = {}  # 存储每个短语的所有事实分数，无论它们是否存在于知识图中
        phrase_weights = np.zeros(len(self.graph.vs["name"]))
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        number_of_occurs = np.zeros(len(self.graph.vs["name"]))

        phrases_and_ids = set()

        # 遍历排名靠前的事实，计算短语节点的初始权重
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = (
                query_fact_scores[top_k_fact_indices[rank]]
                if query_fact_scores.ndim > 0
                else query_fact_scores
            )

            # 关注主语和宾语短语
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    # 如果短语连接到多个段落，则降低其权重（类似于 IDF 的思想）
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(
                            self.ent_node_to_chunk_ids[phrase_key]
                        )

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        # 对短语权重进行平均
        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # 计算每个短语的平均事实分数，用于 linking_score_map
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        # 如果指定了 link_top_k，则只保留分数最高的 k 个短语节点
        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(
                link_top_k, phrase_weights, linking_score_map
            )  # 在此阶段，linking_score_map 的长度由 link_top_k 决定

        # 获取基于稠密检索模型（DPR）的段落分数
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        # 将 DPR 分数分配给段落节点
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)[
                "content"
            ]
            linking_score_map[passage_node_text] = (
                passage_dpr_score * passage_node_weight
            )

        # 将短语权重和段落权重合并为一个数组，作为 PPR 的重置概率（reset probability）
        node_weights = phrase_weights + passage_weights

        # 记录 linking_score_map 中的前 30 个事实（用于调试或日志）
        if len(linking_score_map) > 30:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
            )

        assert (
            sum(node_weights) > 0
        ), f"No phrases found in the graph for the given facts: {top_k_facts}"

        # 基于之前分配的段落和短语权重运行 PPR 算法
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights, damping=self.global_config.damping
        )
        ppr_end = time.time()

        self.ppr_time += ppr_end - ppr_start

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs
        ), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rerank_facts(
        self, query: str, query_fact_scores: np.ndarray
    ) -> Tuple[List[int], List[Tuple], dict]:
        """
        对初步检索到的事实进行重排序（Reranking），以筛选出最相关的事实。

        该函数首先根据初步的相似度分数（`query_fact_scores`）选出候选事实。
        然后，它调用 `rerank_filter` 方法（通常基于更强大的模型或逻辑，如 DSPy）对这些候选事实进行进一步的过滤和排序。
        这有助于提高后续图搜索阶段所使用事实的质量。

        参数:
            query (str): 用于重排序的查询字符串。
            query_fact_scores (np.ndarray): 包含所有事实与查询之间初步相似度分数的数组。

        返回:
            Tuple[List[int], List[Tuple], dict]: 一个元组，包含：
                - top_k_fact_indices (List[int]): 重排序后选出的前 k 个事实在原始事实列表中的索引。
                - top_k_facts (List[Tuple]): 重排序后选出的前 k 个事实本身（三元组形式）。
                - rerank_log (dict): 包含重排序过程信息的字典，如重排序前后的事实列表。
                    - 'facts_before_rerank': 初步筛选出的候选事实列表。
                    - 'facts_after_rerank': 最终选出的事实列表。
        """
        # 加载配置参数：需要保留的链接（事实）数量
        link_top_k: int = self.global_config.linking_top_k

        # 检查是否有事实可供重排序
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

        try:
            # 根据初步分数获取前 k 个事实的索引
            if len(query_fact_scores) <= link_top_k:
                # 如果事实总数少于请求的数量，则使用所有事实
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # 否则，仅获取分数最高的 k 个事实
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
                    ::-1
                ].tolist()

            # 获取实际的事实 ID
            real_candidate_fact_ids = [
                self.fact_node_keys[idx] for idx in candidate_fact_indices
            ]
            # 从嵌入存储中获取事实的详细内容
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            # 解析事实内容（通常存储为字符串形式的三元组）
            candidate_facts = [
                eval(fact_row_dict[id]["content"]) for id in real_candidate_fact_ids
            ]

            # 对候选事实进行重排序（例如使用 LLM 或 DSPy 模块） 选前link_top_k个事实很可能会选中不相关事实
            # TODO：考虑根据计算的向量距离来决定选择前多少个事实
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=link_top_k,
            )

            # 记录重排序前后的事实，用于调试或分析
            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_after_rerank": top_k_facts,
            }

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return (
                [],
                [],
                {"facts_before_rerank": [], "facts_after_rerank": [], "error": str(e)},
            )

    def run_ppr(
        self, reset_prob: np.ndarray, damping: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None:
            damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

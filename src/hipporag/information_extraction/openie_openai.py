import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    """
    定义文档块信息的类型结构。
    """

    num_tokens: int  # 块中的 token 数量
    content: str  # 块的文本内容
    chunk_order: List[Tuple]  # 块的顺序信息
    full_doc_ids: List[str]  # 所属完整文档的 ID 列表


@dataclass
class LLMInput:
    """
    定义传递给 LLM 的输入数据结构。
    """

    chunk_id: str  # 块的唯一标识符
    input_message: List[Dict]  # 发送给 LLM 的消息列表（包含 prompt）


def _extract_ner_from_response(real_response):
    """
    从 LLM 的原始响应文本中提取命名实体列表。
    使用正则表达式查找 JSON 格式的 "named_entities" 字段。
    """
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        # 如果未匹配到模式，则返回空列表
        return []
    return eval(match.group())["named_entities"]


class OpenIE:
    """
    基于 OpenAI 接口的大语言模型进行开放信息抽取（OpenIE）的类。
    主要功能包括命名实体识别（NER）和三元组抽取。
    """

    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        # 初始化提示模板管理器，用于管理系统、用户和助手的 prompt
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        """
        对给定的文本段落执行命名实体识别（NER）。

        参数:
            chunk_key (str): 文本块的唯一标识符。
            passage (str): 待处理的文本内容。

        返回:
            NerRawOutput: 包含识别出的实体列表及元数据的对象。
        """
        # PREPROCESSING
        # 渲染 NER 提示模板
        ner_input_message = self.prompt_template_manager.render(
            name="ner", passage=passage
        )
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            # 调用 LLM 进行推理
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata["cache_hit"] = cache_hit

            # 如果因为长度原因截断，尝试修复 JSON
            if metadata["finish_reason"] == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            # 从响应中提取实体并去重
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            # 捕获异常并记录，返回包含错误信息的对象
            logger.warning(e)
            metadata.update({"error": str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata,  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata,
        )

    def triple_extraction(
        self, chunk_key: str, passage: str, named_entities: List[str]
    ) -> TripleRawOutput:
        """
        基于给定的文本段落和已识别的实体，抽取三元组（主语，谓语，宾语）。

        参数:
            chunk_key (str): 文本块的唯一标识符。
            passage (str): 文本内容。
            named_entities (List[str]): 该段落中已识别的命名实体列表。

        返回:
            TripleRawOutput: 包含抽取出的三元组列表及元数据的对象。
        """

        def _extract_triples_from_response(real_response):
            """内部辅助函数：从响应中提取三元组 JSON"""
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            return eval(match.group())["triples"]

        # PREPROCESSING
        # 渲染三元组抽取提示模板，传入文本和实体列表
        messages = self.prompt_template_manager.render(
            name="triple_extraction",
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities}),
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            # 调用 LLM 进行推理
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata["cache_hit"] = cache_hit

            # 处理截断情况
            if metadata["finish_reason"] == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            # 提取并过滤无效三元组
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({"error": str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key, response=raw_response, metadata=metadata, triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets,
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        """
        对单个文本块执行完整的 OpenIE 流程（NER + 三元组抽取）。
        """
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key,
            passage=passage,
            named_entities=ner_output.unique_entities,
        )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(
        self, chunks: Dict[str, ChunkInfo]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.
        使用多线程同步执行批量 OpenIE，包括 NER 和三元组抽取。

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.
            chunks (Dict[str, ChunkInfo]): 待并入图中的文本块。键是哈希后的块 ID，值是块信息。

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
                - 一个字典，键为块 ID，值为 NER 结果实例。
                - 一个字典，键为块 ID，值为三元组抽取结果实例。
        """

        # Extract passages from the provided chunks
        # 从提供的块信息中提取文本内容
        chunk_passages = {
            chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()
        }

        # 用于存储所有并行执行的 NER（命名实体识别）任务的结果
        ner_results_list = []
        # 累计所有并发请求中输入提示词（Prompt）所消耗的 Token 总数
        total_prompt_tokens = 0
        # 累计所有并发请求中模型生成（Completion）所消耗的 Token 总数
        total_completion_tokens = 0
        # 统计在这一批次请求中，有多少次请求是直接命中了本地缓存
        num_cache_hit = 0

        # 第一阶段：并行执行 NER
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Create NER futures for each chunk
            # 为每个块创建 NER 任务
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                # 更新统计指标
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1

                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0

        # 第二阶段：并行执行三元组抽取（依赖 NER 结果）
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            # 为每个块创建三元组抽取任务
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            # 收集结果
            pbar = tqdm(
                as_completed(re_futures),
                total=len(re_futures),
                desc="Extracting triples",
            )
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict

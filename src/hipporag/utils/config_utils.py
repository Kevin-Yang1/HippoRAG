import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """唯一的配置类。"""
    # LLM 特定属性
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "指示使用哪个 LLM 模型的类名。"}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "LLM 模型的基础 URL，如果为 None，则使用 OPENAI 服务。"}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "OpenAI 兼容嵌入模型的基础 URL，如果为 None，则使用 OPENAI 服务。"}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "LLM 模型的 Azure 端点 URI，如果为 None，直接使用 OPENAI 服务。"}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "OpenAI 嵌入模型的 Azure 端点 URI，如果为 None，直接使用 OPENAI 服务。"}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "每次推理生成的最大新 token 数。"}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "为每个输入消息生成多少个聊天完成选项。"}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "随机种子。"}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "每次推理的采样温度。"}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "指定模型必须输出的格式。"}
    )
    
    ## LLM 特定属性 -> 异步超参数
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "异步 API 调用的最大重试次数。"}
    )
    # 存储特定属性
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "如果设置为 True，将忽略所有现有的 openie 文件并从头开始重建。"}
    )

    # 存储特定属性 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "如果设置为 True，将忽略所有现有的存储文件和图数据并从头开始重建。"}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "重排序 dspy 文件的路径。"}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "在 PPR 中修改段落节点权重的乘法因子。"}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "如果设置为 True，将把 OpenIE 模型保存到磁盘。"}
    )
    
    # 预处理特定属性
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "预处理中使用的基于文本的预处理器名称。"}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "预处理中使用的编码器名称（目前专门用于文档分块）。"}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "相邻块之间的重叠 token 数。"}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "每个块可以包含的最大 token 数。如果设置为 None，整个文档将被视为单个块。"}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # 信息抽取特定属性
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "指示使用哪个信息抽取模型的类名。"}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "使用的 OpenIE 模型模式。"}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "是否跳过图构建。首次运行 vllm 离线索引时设置为 true。"}
    )
    
    
    # 嵌入特定属性
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "指示使用哪个嵌入模型的类名。"}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "调用嵌入模型的批次大小。"}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "是否归一化编码后的嵌入。"}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "嵌入模型的最大序列长度。"}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "本地嵌入模型的数据类型。"}
    )
    
    
    
    # 图构建特定属性
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "构建同义词边时 knn 检索的 k 值。"}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "构建同义词边时 knn 检索的查询嵌入批次大小。"}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "构建同义词边时 knn 检索的键嵌入批次大小。"}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "包含候选同义词节点的相似度阈值。"}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "图是否有向。"}
    )
    
    
    
    # 检索特定属性
    linking_top_k: int = field(
        default=5,
        metadata={"help": "每个检索步骤中链接节点的数量"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "每个步骤检索 k 个文档"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "ppr 算法的阻尼因子。"}
    )
    
    
    # QA 特定属性
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "对于回答单个问题，我们用于交错检索和推理的最大步骤数。"}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "提供给 QA 模型阅读的前 k 个文档。"}
    )
    
    # 保存目录（最高级目录）
    save_dir: str = field(
        default=None,
        metadata={"help": "保存所有相关信息的目录。如果给出，将覆盖所有默认的 save_dir 设置。如果未给出，且我们没有运行特定数据集，则默认为 `outputs`，否则默认为数据集自定义的输出目录。"}
    )
    
    
    
    # 数据集运行特定属性
    ## 数据集运行特定属性 -> 通用
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "要使用的数据集。如果指定，意味着我们将运行特定数据集。如果未指定，意味着我们自由运行。"}
    )
    ## 数据集运行特定属性 -> 图
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "实验中使用的图类型。"}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "要使用的语料库长度。"}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # 如果未给出 save_dir
            if self.dataset is None: self.save_dir = 'outputs' # 自由运行
            else: self.save_dir = os.path.join('outputs', self.dataset) # 在此自定义数据集的输出目录
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")

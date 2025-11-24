# HippoRAG AI 编码指引（面向智能代理）

**目标**：让 AI 代理在本仓库中即刻高效工作，了解架构、工作流、项目约定与扩展点。内容仅基于仓库可发现事实，不包含愿景性条目。

**大图景架构**
- `HippoRAG`（`src/hipporag/HippoRAG.py`）：核心入口，编排索引与检索流程。
- `EmbeddingStore`（`src/hipporag/embedding_store.py`）：管理段落、实体、事实三类向量，落盘为 Parquet（文件名模式：`vdb_{namespace}.parquet`）。
- OpenIE（`src/hipporag/information_extraction/`）：从文本抽取实体与三元组。含在线 OpenAI 版与离线 `Transformers`/`vLLM` 版。
- LLM 抽象（`src/hipporag/llm/base.py`）与向量抽象（`src/hipporag/embedding_model/base.py`）：统一推理与编码接口。
- 图与检索：基于 `igraph` 将“段落/实体”作为节点，“事实/共现”为边；检索阶段结合稠密得分与 PPR 重排。

**关键数据流**
- 索引 `HippoRAG.index(docs)`：
    - OpenIE：`ner` + 三元组抽取；支持 `openie_mode=='offline'` 的预抽取与结果缓存（可由 `save_openie` 控制持久化）。
    - 向量化：段落、实体、事实分别编码，写入各自 `EmbeddingStore`（命名空间区分）。
    - 图构建：添加事实边、段落-实体边、同义边，最终保存 `igraph`。
- 检索 `HippoRAG.retrieve(queries, num_to_retrieve)`：
    - 事实检索 → 识别记忆过滤 → 段落稠密打分 → PPR 重排；若长查询无相关事实则回退稠密检索。

**开发工作流**
- 环境与安装（Python 3.10）：
    ```bash
    conda create -n hipporag python=3.10
    conda activate hipporag
    pip install -r requirements.txt
    ```
- 运行脚本/测试（根目录均为集成测试脚本）：
    ```bash
    # 使用 OpenAI（需 env: OPENAI_API_KEY）
    python tests_openai.py
    # 使用本地/离线模型
    python tests_local.py
    # 使用 Azure
    python tests_azure.py
    ```
- 快速示例（OpenAI 在线）：见 `README.md` 与 `demo_openai.py`；本地/离线参见 `demo_local.py`、`information_extraction/openie_transformers_offline.py`。

**项目约定与模式**
- 类型：所有函数参数/返回值使用类型注解。
- 日志：仅用 `src.hipporag.utils.logging_utils.get_logger`，禁止 `print`。
- 路径：统一使用 `os.path.join`；数据默认写入 `save_dir` 子目录（由 `HippoRAG(save_dir=...)` 控制）。
- 存储：向量表使用 Parquet；三类向量通过命名空间隔离；OpenIE 结果可缓存到 `outputs/...`。
- Rerank：`src/hipporag/rerank.py` 支持基于 DSPy 的事实过滤，路径由 `global_config.rerank_dspy_file_path` 指定。

**扩展点（按已实现接口）**
- 新 LLM：继承 `BaseLLM`，实现 `infer`/`ainfer`/`batch_infer`；参考 `llm/openai_gpt.py` 与 `llm/transformers_offline.py`。
- 新向量模型：继承 `BaseEmbeddingModel`，实现 `batch_encode`；参考 `embedding_model/NVEmbedV2.py`/`GritLM.py`/`Contriever.py`。
- 新 OpenIE：实现与 `OpenIE` 兼容接口（至少 `ner` 与三元组抽取方法）；参考 `information_extraction/openie_openai.py`。

**配置与可调参数**
- 入口：`HippoRAG(save_dir=..., llm_model_name=..., embedding_model_name=...)`；更细粒度见 `src/hipporag/utils/config_utils.py`。
- 常用字段：`llm_name`、`embedding_model_name`、`openie_mode`（online/offline）、`save_openie`、`rerank_dspy_file_path`、Azure/OpenAI 相关 endpoint/key。

**代码位置速览**
- 核心：`src/hipporag/HippoRAG.py`、`src/hipporag/StandardRAG.py`
- 提示词：`src/hipporag/prompts/`（包含 `templates/ner.py`、`templates/triple_extraction.py` 等）
- 评测：`src/hipporag/evaluation/`（检索/问答评测脚本）
- 复现实验：`reproduce/` 与根目录多个 `demo_*.py`/`main*.py`

如需我补充特定配置字段说明、离线模型最小可运行例子或图保存/加载细节，请指出不清楚的部分，我再迭代完善。

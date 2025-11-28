"""
此模块定义了用于 Embedding 模型的特定任务指令（Instructions）。

在 HippoRAG 中，使用不同的指令来指导 Embedding 模型针对特定的检索任务生成向量表示。
例如，检索事实（三元组）和检索段落（文档）需要不同的语义侧重。
通过提供明确的任务描述（如 "Given a question, retrieve relevant triplet facts..."），
可以显著提高基于指令微调的 Embedding 模型（如 Instructor, NV-Embed 等）的检索性能。
"""


def get_query_instruction(linking_method):
    instructions = {
        "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
        "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
        "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
        "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
        "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
    }
    default_instruction = (
        "Given a question, retrieve relevant documents that best answer the question."
    )
    return instructions.get(linking_method, default_instruction)

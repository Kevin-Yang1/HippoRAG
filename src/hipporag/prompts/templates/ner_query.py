"""
此模板用于从用户查询 (Query) 中提取关键命名实体。
它指示模型识别出对于回答问题至关重要的实体，并以 JSON 格式返回。
"""

ner_system = """You're a very effective entity extraction system.
"""

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
# 单样本演示的输入：包含指令和一个示例问题。

query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""
# 单样本演示的输出：提取出的实体列表。

# query_prompt_template = """
# Question: {}

# """
prompt_template = [
    # 最终的提示词模板结构。
    # 包含系统指令、单样本演示以及用户实际查询的占位符。
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "Question: ${query}"},
]

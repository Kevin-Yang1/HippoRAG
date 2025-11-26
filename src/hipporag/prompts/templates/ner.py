"""
此模板用于命名实体识别 (NER) 任务。
它指示模型从给定的段落中提取命名实体，并以 JSON 列表格式返回。
"""

ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""
# 单样本演示的输入段落。


one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""
# 单样本演示的期望输出（JSON 格式的实体列表）。


prompt_template = [
    # 最终的提示词模板结构。
    # 包含系统指令、单样本演示（输入和输出）以及用户实际输入的占位符。
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"},
]

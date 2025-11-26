"""
此模板用于三元组抽取 (Triple Extraction) 任务。
它指示模型基于给定的段落和已识别的命名实体列表，构建 RDF 图（三元组列表）。
要求三元组中的实体尽量来自提供的实体列表，并解析代词。
"""

from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""
# 系统指令：定义任务目标（构建 RDF 图）、输出格式（JSON 三元组列表）和约束条件（使用给定实体、解析代词）。


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""
# 输入框架模板：将段落和实体列表组合成特定的输入格式。


ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output
)
# 单样本演示的输入：使用 NER 模板中的示例数据填充框架。


ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""
# 单样本演示的输出：期望的三元组列表。


prompt_template = [
    # 最终的提示词模板结构。
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {
        "role": "user",
        "content": convert_format_to_template(
            original_string=ner_conditioned_re_frame,
            placeholder_mapping=None,
            static_values=None,
        ),
    },
]

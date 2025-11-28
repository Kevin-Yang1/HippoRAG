import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt


class Fact(BaseModel):
    # 定义事实的数据模型：一个包含多个事实的列表，每个事实是由3个字符串组成的列表 [主语, 谓语, 宾语]  注意：并没有强约束，只是约定
    fact: list[list[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


class DSPyFilter:
    def __init__(self, hipporag):
        """
        初始化对象，配置处理输入和输出消息所需的配置和模板。

        参数:
        hipporag : 提供全局配置和推理所需 LLM 模型的对象。

        属性:
        dspy_file_path : 全局配置中指定的用于重排序的文件路径。
        one_input_template : 用于格式化输入消息的字符串模板，包含特定字段的占位符。
        one_output_template : 用于格式化输出消息的字符串模板，包含特定字段。
        message_template : 使用指定的 dspy 文件路径生成的模板。
        llm_infer_fn : 用于使用提供的 LLM 模型进行推理的函数引用。
        model_name : 全局配置中指定的语言模型名称。
        default_gen_kwargs : 用于存储默认生成关键字参数的字典。
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        # 定义单个输入的模板，包含问题和过滤前的事实
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        # 定义单个输出的模板，包含过滤后的事实
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        # 生成完整的消息模板（包含系统提示词和示例）
        self.message_template = self.make_template(dspy_file_path)
        # 获取 LLM 推理函数
        self.llm_infer_fn = hipporag.llm_model.infer
        # 获取模型名称
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        """
        根据 DSPy 文件路径或默认提示词构建消息模板。
        """
        # 如果提供了路径则加载文件，否则使用默认的最佳 DSPy 提示词
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, "r"))
        else:
            dspy_saved = best_dspy_prompt

        # 获取系统提示词
        system_prompt = dspy_saved["prog"]["system"]
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        # 获取示例（demos）并添加到模板中
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append(
                {
                    "role": "user",
                    "content": self.one_input_template.format(
                        question=demo["question"],
                        fact_before_filter=demo["fact_before_filter"],
                    ),
                }
            )
            message_template.append(
                {
                    "role": "assistant",
                    "content": self.one_output_template.format(
                        fact_after_filter=demo["fact_after_filter"]
                    ),
                }
            )
        return message_template

    def parse_filter(self, response):
        """
        解析 LLM 的响应，提取过滤后的事实。
        """
        sections = [(None, [])]
        # 定义字段头的正则表达式模式
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")

        # 逐行解析响应，按字段头分割内容
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        # 合并每个部分的内容
        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    # 验证解析后的值是否符合 Fact 模型
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, fact_before_filter):
        """
        调用 LLM 进行推理。
        """
        # 复制模板并添加当前的用户输入
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(
                    question=question, fact_before_filter=fact_before_filter
                ),
            }
        )

        # 设置最大生成 token 数
        self.default_gen_kwargs["max_completion_tokens"] = 512

        # 调用 LLM 推理函数
        response = self.llm_infer_fn(
            messages=messages, model=self.model_name, **self.default_gen_kwargs
        )

        # 如果响应包含多个选项，返回第一个
        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(
        self,
        query: str,
        candidate_items: List[Tuple],
        candidate_indices: List[int],
        len_after_rerank: int = None,
    ) -> Tuple[List[int], List[Tuple], dict]:
        """
        对候选事实进行重排序。

        参数:
            query: 查询字符串。
            candidate_items: 候选事实列表（元组形式）。
            candidate_indices: 候选事实的原始索引列表。
            len_after_rerank: 重排序后保留的事实数量。

        返回:
            排序后的索引列表, 排序后的事实列表, 包含置信度的字典。
        """
        # 准备输入数据格式
        fact_before_filter = {
            "fact": [list(candidate_item) for candidate_item in candidate_items]
        }
        try:
            # 调用 LLM 获取过滤后的事实
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print("exception", e)
            generated_facts = []

        result_indices = []
        # 将生成的事实映射回原始候选列表中的索引
        for generated_fact in generated_facts:
            # 使用 difflib 找到最匹配的原始事实（处理可能的生成微小差异）
            closest_matched_fact = difflib.get_close_matches(
                str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0
            )[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print("result_indices exception", e)

        # 根据结果索引重新排序候选索引和项目
        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]

        # 返回截断后的结果
        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )

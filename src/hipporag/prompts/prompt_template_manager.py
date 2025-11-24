import os
import asyncio
import importlib.util
from string import Template
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field, asdict


from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    """
    提示模板管理器，用于加载、管理和渲染 LLM 的提示词模板。
    支持从指定目录加载 Python 脚本定义的模板，并处理角色映射。
    """

    # templates_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Directory containing template scripts. Default to the `templates` dir under dir whether this class is defined."}
    # )
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        },
        metadata={
            "help": "Mapping from default roles in prompte template files to specific LLM providers' defined roles."
        },
        # 角色映射：将模板文件中的默认角色（如 system, user）映射到特定 LLM 提供商定义的角色名称
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "help": "A dict from prompt template names to templates. A prompt template can be a Template instance or a chat history which is a list of dict with content as Template instance."
        },
        # 模板字典：键为模板名称，值为 Template 实例或包含 Template 的聊天记录列表
    )

    def __post_init__(self) -> None:
        """
        Initialize the templates directory and load templates.
        初始化模板目录并加载所有模板。
        """
        # if self.templates_dir is None:
        #     current_file_path = os.path.abspath(__file__)
        #     package_dir = os.path.dirname(current_file_path)
        #     self.templates_dir = os.path.join(package_dir, "templates")
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)

        # abs path to dir where each *.py file (exclude __init__.py) contains a variable prompt_template (a str or a chat history with content as raw str for being converted to a Template)
        # 获取模板目录的绝对路径，该目录下的每个 .py 文件（除 __init__.py 外）都应包含一个 prompt_template 变量
        self.templates_dir = os.path.join(package_dir, "templates")

        self._load_templates()

    def _load_templates(self) -> None:
        """
        Load all templates from Python scripts in the templates directory.
        从模板目录中的 Python 脚本加载所有模板。
        """
        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory '{self.templates_dir}' does not exist.")
            raise FileNotFoundError(
                f"Templates directory '{self.templates_dir}' does not exist."
            )

        logger.info(f"Loading templates from directory: {self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]

                try:
                    # 尝试动态导入模板模块
                    try:
                        module_name = f"src.hipporag.prompts.templates.{script_name}"
                        module = importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        module_name = f".prompts.templates.{script_name}"
                        module = importlib.import_module(module_name, "hipporag")

                    # spec = importlib.util.spec_from_file_location(script_name, script_path)
                    # module = importlib.util.module_from_spec(spec)
                    # spec.loader.exec_module(module)

                    if not hasattr(module, "prompt_template"):
                        logger.error(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )
                        raise AttributeError(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )

                    prompt_template = module.prompt_template
                    logger.debug(f"Loaded template from {module_name}")

                    # 处理不同类型的模板格式
                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item
                        for item in prompt_template
                    ):
                        # Adjust roles based on the provided role mapping
                        # 如果是列表（聊天记录格式），则根据角色映射调整角色，并将内容转换为 Template 对象
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(
                                item["role"], item["role"]
                            )
                            item["content"] = (
                                item["content"]
                                if isinstance(item["content"], Template)
                                else Template(item["content"])
                            )
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(
                            f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict]."
                        )

                    logger.debug(
                        f"Successfully loaded template '{script_name}' from '{module_name}.py'."
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load template from '{module_name}.py': {e}"
                    )
                    raise

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Render a template with the provided variables.
        使用提供的变量渲染指定的模板。

        Args:
            name (str): The name of the template. 模板名称。
            kwargs: Placeholder values for the template. 模板中的占位符变量。

        Returns:
            Union[str, List[Dict[str, Any]]]: The rendered template or chat history. 渲染后的字符串或聊天记录列表。

        Raises:
            ValueError: If a required variable is missing. 如果缺少必要的变量。
        """
        template = self.get_template(name)
        if isinstance(template, Template):
            # Render a single string template
            # 渲染单个字符串模板
            try:
                result = template.substitute(**kwargs)
                logger.debug(
                    f"Successfully rendered template '{name}' with variables: {kwargs}."
                )
                return result
            except KeyError as e:
                logger.error(f"Missing variable for template '{name}': {e}")
                raise ValueError(f"Missing variable for template '{name}': {e}")
        elif isinstance(template, list):
            # Render a chat history
            # 渲染聊天记录模板（列表格式）
            try:
                rendered_list = [
                    {
                        "role": item["role"],
                        "content": item["content"].substitute(**kwargs),
                    }
                    for item in template
                ]
                logger.debug(
                    f"Successfully rendered chat history template '{name}' with variables: {kwargs}."
                )
                return rendered_list
            except KeyError as e:
                logger.error(f"Missing variable in chat history template '{name}': {e}")
                raise ValueError(
                    f"Missing variable in chat history template '{name}': {e}"
                )

    def sync_render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        同步执行渲染（虽然 render 本身不是异步的，但这里使用了 asyncio.run，可能是为了兼容某些异步上下文或未来扩展）。
        """
        return asyncio.run(self.render(name, **kwargs))

    def list_template_names(self) -> List[str]:
        """
        List all available template names.
        列出所有可用的模板名称。

        Returns:
            List[str]: A list of template names. 模板名称列表。
        """
        logger.info("Listing all available template names.")

        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        Retrieve a template by name.
        根据名称获取模板对象。

        Args:
            name (str): The name of the template. 模板名称。

        Returns:
            Union[Template, List[Dict[str, Any]]]: The requested template. 请求的模板对象。

        Raises:
            KeyError: If the template is not found. 如果模板未找到。
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found.")
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieved template '{name}'.")

        return self.templates[name]

    def print_template(self, name: str) -> None:
        """
        Print the prompt template string or chat history structure for the given template name.
        打印指定模板名称的提示词模板字符串或聊天记录结构。

        Args:
            name (str): The name of the template. 模板名称。

        Raises:
            KeyError: If the template is not found. 如果模板未找到。
        """
        try:
            template = self.get_template(name)
            print(f"Template name: {name}")
            if isinstance(template, Template):
                print(template.template)
            elif isinstance(template, list):
                for item in template:
                    print(f"Role: {item['role']}, Content: {item['content']}")
            logger.info(f"Printed template '{name}'.")
        except KeyError as e:
            logger.error(f"Failed to print template '{name}': {e}")
            raise

    def is_template_name_valid(self, name: str) -> bool:
        """
        检查模板名称是否有效（是否存在）。
        """
        return name in self.templates

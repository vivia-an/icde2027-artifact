import os
import re
import yaml
from autogen import LLMConfig
from pydantic import BaseModel

# 禁用系统代理，防止连接错误
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'



class ModelConfig(BaseModel):
    name: str
    model: str
    api_key: str
    type: str
    base_url: str | None = None  # 支持 base_url 字段（可选）


class Config(BaseModel):
    specializations: dict[str, str]
    models: dict[str, ModelConfig]

    def get_model(self, specialization: str):
        model = self.specializations.get(specialization)
        return self.models.get(model)


ENV_PATTERN = re.compile(r"\$\{([^}^{]+)\}")
def resolve_env_vars(value):
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    elif isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            env_var = match.group(1)
            # Missing env → empty string (do not ship secrets in yaml)
            return os.environ.get(env_var) or ""

    return value


CONFIG_CACHE = None
def load_config() -> Config:
    global CONFIG_CACHE
    if CONFIG_CACHE:
        return CONFIG_CACHE
    
    llm_config_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(llm_config_file_dir, "llm_config.yaml")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        config = resolve_env_vars(config)
    
    config = Config(**config)
    CONFIG_CACHE = config
    return config


def get_llm_config(model_specialization, **kwargs):
    """
    获取 autogen 的 LLM 配置
    
    autogen 使用 config_list 格式来配置 OpenAI 兼容的 API
    """
    config = load_config()
    model = config.get_model(model_specialization)
    
    # Env overrides yaml so local keys can stay in yaml while open-source uses env.
    api_key = model.api_key
    if model.base_url and "deepseek" in (model.base_url or "").lower():
        api_key = os.environ.get("DEEPSEEK_API_KEY") or api_key
    elif (model.model or "").startswith("deepseek"):
        api_key = os.environ.get("DEEPSEEK_API_KEY") or api_key
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or api_key

    # 构建 config_list 中的单个配置项
    model_config = {
        "model": model.model,
        "api_key": api_key,
    }
    
    # 如果有 base_url，添加到配置中（用于 OpenAI 兼容的 API，如 DeepSeek）
    if model.base_url:
        model_config["base_url"] = model.base_url

    # DeepSeek V4：开 thinking + high effort（最优挖掘质量）。
    # ag2 需配合 ag2_deepseek_thinking_patch 回传 reasoning_content，否则工具环可能 400。
    if model.model.startswith("deepseek-v4"):
        effort = os.environ.get("SDC_REASONING_EFFORT", "high")
        model_config["extra_body"] = {"thinking": {"type": "enabled"}}
        model_config["reasoning_effort"] = effort
    
    # 构建完整的 llm_config
    llm_config_params = {
        "config_list": [model_config],
        **kwargs
    }
    
    return LLMConfig(**llm_config_params)



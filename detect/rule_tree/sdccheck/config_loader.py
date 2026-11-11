"""配置加载器 - 复用agents中的配置系统"""

import os
import sys
import re
import yaml
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """模型配置"""
    name: str
    model: str
    api_key: str
    type: str
    timeout: int = 3600
    base_url: Optional[str] = None


class Config(BaseModel):
    """全局配置"""
    specializations: Dict[str, str]
    models: Dict[str, ModelConfig]

    def get_model(self, specialization: str) -> Optional[ModelConfig]:
        """根据specialization获取模型配置"""
        model_name = self.specializations.get(specialization)
        if model_name:
            return self.models.get(model_name)
        return None


ENV_PATTERN = re.compile(r"\$\{([^}^{]+)\}")

def resolve_env_vars(value):
    """解析环境变量"""
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    elif isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            env_var = match.group(1)
            env_value = os.environ.get(env_var)
            return env_value
    return value


CONFIG_CACHE = None

def load_config() -> Config:
    """加载配置（使用agents的配置文件）"""
    global CONFIG_CACHE
    if CONFIG_CACHE:
        return CONFIG_CACHE
    
    # 尝试找到 llm_config.yaml
    current_dir = Path(__file__).parent
    
    # 可能的配置文件路径（按优先级排序）
    possible_paths = [
        # 安装后路径 (site-packages/config/llm_config.yaml)
        current_dir.parent / "config" / "llm_config.yaml",
        # 开发模式路径 (sdccheck/agents/llm_config.yaml)
        current_dir.parent / "agents" / "llm_config.yaml",
        # sdccheck 包内路径
        current_dir / "config" / "llm_config.yaml",
        # 相对路径 (当前工作目录)
        Path("agents/llm_config.yaml"),
        Path("config/llm_config.yaml"),
    ]
    
    config_file = None
    for path in possible_paths:
        if path.exists():
            config_file = path
            break
    
    if not config_file:
        raise FileNotFoundError(
            f"未找到 llm_config.yaml 配置文件。尝试的路径：\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    with open(config_file, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
        config_data = resolve_env_vars(config_data)
    
    config = Config(**config_data)
    CONFIG_CACHE = config
    return config


def get_model_config(specialization: str = "default") -> ModelConfig:
    """
    获取模型配置
    
    参数:
        specialization: 专门化类型，如 'default', 'test', 'conversation' 等
    
    返回:
        ModelConfig: 模型配置对象
    """
    config = load_config()
    model_config = config.get_model(specialization)
    
    if not model_config:
        # 尝试使用默认配置
        model_config = config.get_model("default")
    
    if not model_config:
        raise ValueError(
            f"未找到 specialization '{specialization}' 的配置，"
            f"可用的配置: {list(config.specializations.keys())}"
        )
    
    return model_config


def list_available_specializations():
    """列出所有可用的specialization"""
    config = load_config()
    return list(config.specializations.keys())


def list_available_models():
    """列出所有可用的模型"""
    config = load_config()
    return list(config.models.keys())


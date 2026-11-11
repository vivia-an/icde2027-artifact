"""LLM配置相关类和常量"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """从字典创建配置"""
        return cls(**data)


# 预定义配置
PREDEFINED_CONFIGS = {
    "openai_gpt4": LLMConfig(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=2000
    ),
    "openai_gpt35": LLMConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=2000
    ),
    "anthropic_claude": LLMConfig(
        model_name="claude-3-sonnet-20240229",
        temperature=0.1,
        max_tokens=2000
    ),
    "ollama_llama2": LLMConfig(
        model_name="llama2",
        temperature=0.1,
        max_tokens=2000,
        base_url="http://localhost:11434"
    ),
    "deepseek_v3": LLMConfig(
        model_name="deepseek-chat",
        temperature=0.1,
        max_tokens=2000,
        base_url="https://api.deepseek.com"
    ),
    "mock": LLMConfig(
        model_name="mock-model",
        temperature=0.1,
        max_tokens=2000
    )
}


def get_config(name: str) -> LLMConfig:
    """获取预定义配置"""
    if name not in PREDEFINED_CONFIGS:
        raise ValueError(f"未知配置: {name}，可用配置: {list(PREDEFINED_CONFIGS.keys())}")
    return PREDEFINED_CONFIGS[name]
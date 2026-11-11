"""LLM提供商配置和客户端实现"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import os
import logging
from .config import LLMConfig

try:
    from .config_loader import get_model_config, ModelConfig
    USE_AGENTS_CONFIG = True
except ImportError:
    USE_AGENTS_CONFIG = False


class LLMProvider(ABC):
    """LLM提供商抽象基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"SDCCheck.{self.__class__.__name__}")
        # Token 使用统计（每次调用后更新）
        self.last_token_usage: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    
    @abstractmethod
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """聊天完成接口"""
        pass
    
    def chat_completion_with_usage(self, prompt: str, system_prompt: Optional[str] = None,
                                   temperature: float = 0.1, max_tokens: int = 2000) -> Tuple[str, Dict[str, Any]]:
        """
        聊天完成接口（带 token 使用统计）
        
        返回:
            Tuple[str, Dict]: (响应内容, token 使用统计)
        """
        response = self.chat_completion(prompt, system_prompt, temperature, max_tokens)
        return response, self.last_token_usage.copy()
    
    def get_last_token_usage(self) -> Dict[str, Any]:
        """获取上次调用的 token 使用统计"""
        return self.last_token_usage.copy()
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提供商是否可用"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI提供商实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            
            api_key = self.config.api_key or os.getenv("UBI_API_KEY")
            if not api_key:
                self.logger.warning("未找到OpenAI API密钥")
                return
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            self.logger.info("OpenAI客户端初始化成功")
            
        except ImportError:
            self.logger.error("未安装openai库，请运行: pip install openai")
        except Exception as e:
            self.logger.error(f"初始化OpenAI客户端失败: {e}")
    
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """OpenAI聊天完成"""
        if not self.client:
            raise RuntimeError("OpenAI客户端未初始化")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取 token 使用信息
            if hasattr(response, 'usage') and response.usage:
                self.last_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self.logger.debug(f"Token 使用: {self.last_token_usage}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查OpenAI是否可用"""
        return self.client is not None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude提供商实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                self.logger.warning("未找到Anthropic API密钥")
                return
            
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.config.timeout
            )
            
            self.logger.info("Anthropic客户端初始化成功")
            
        except ImportError:
            self.logger.error("未安装anthropic库，请运行: pip install anthropic")
        except Exception as e:
            self.logger.error(f"初始化Anthropic客户端失败: {e}")
    
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Anthropic聊天完成"""
        if not self.client:
            raise RuntimeError("Anthropic客户端未初始化")
        
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 提取 token 使用信息（Anthropic 格式）
            if hasattr(response, 'usage') and response.usage:
                self.last_token_usage = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
                }
                self.logger.debug(f"Token 使用: {self.last_token_usage}")
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic API调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查Anthropic是否可用"""
        return self.client is not None


class OllamaProvider(LLMProvider):
    """Ollama本地模型提供商实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化Ollama客户端"""
        try:
            import requests
            
            # 测试连接
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.client = requests.Session()
                self.logger.info(f"Ollama客户端初始化成功，地址: {self.base_url}")
            else:
                self.logger.warning(f"无法连接到Ollama服务: {self.base_url}")
                
        except ImportError:
            self.logger.error("未安装requests库，请运行: pip install requests")
        except Exception as e:
            self.logger.error(f"初始化Ollama客户端失败: {e}")
    
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Ollama聊天完成"""
        if not self.client:
            raise RuntimeError("Ollama客户端未初始化")
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise RuntimeError(f"Ollama API返回错误: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Ollama API调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查Ollama是否可用"""
        return self.client is not None


class MockProvider(LLMProvider):
    """模拟提供商，用于测试和演示"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
    
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """模拟聊天完成"""
        # 根据prompt内容返回不同的模拟响应
        if "约束生成" in prompt or "constraint" in prompt.lower():
            response = self._mock_constraint_response()
        elif "SQL" in prompt or "sql" in prompt.lower():
            response = self._mock_sql_response()
        elif "分析" in prompt or "analysis" in prompt.lower():
            response = self._mock_analysis_response()
        else:
            response = '{"result": "模拟响应", "status": "success"}'
        
        # 模拟 token 使用（基于实际输入输出长度估算）
        prompt_tokens = len(prompt) // 4  # 粗略估算：4字符约等于1 token
        system_tokens = len(system_prompt) // 4 if system_prompt else 0
        completion_tokens = len(response) // 4
        
        self.last_token_usage = {
            "prompt_tokens": prompt_tokens + system_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + system_tokens + completion_tokens
        }
        self.logger.debug(f"Mock Token 使用: {self.last_token_usage}")
        
        return response
    
    def is_available(self) -> bool:
        """模拟提供商总是可用"""
        return True
    
    def _mock_constraint_response(self) -> str:
        """模拟约束生成响应"""
        return '''
        {
            "constraints": [
                {
                    "name": "DP权重一致性",
                    "description": "在数据并行配置中，相同模型部分的权重在各DP-rank间应保持一致",
                    "type": "consistency",
                    "logic": "GROUP BY name, tp -> (COUNT(DISTINCT dp) > 1) IMPLIES (COUNT(DISTINCT cksum) = 1)",
                    "tables": ["coredump"],
                    "params": {}
                }
            ],
            "reasoning": "基于数据并行配置生成的一致性约束"
        }
        '''
    
    def _mock_sql_response(self) -> str:
        """模拟SQL生成响应"""
        return '''
        {
            "sql": "SELECT JSON_EXTRACT(data, '$.name') as name, JSON_EXTRACT(data, '$.tp') as tp, COUNT(DISTINCT JSON_EXTRACT(data, '$.dp')) as dp_count, COUNT(DISTINCT JSON_EXTRACT(data, '$.cksum')) as cksum_count FROM coredump GROUP BY name, tp HAVING dp_count > 1 AND cksum_count > 1",
            "explanation": "此SQL用于检查DP权重一致性约束，从JSON数据中提取相关字段"
        }
        '''
    
    def _mock_analysis_response(self) -> str:
        """模拟结果分析响应"""
        return '''
        {
            "status": "pass",
            "summary": "通过: 未发现违反约束的情况",
            "violations": [],
            "suggestion": null,
            "confidence": 0.95
        }
        '''


class DeepSeekProvider(LLMProvider):
    """DeepSeek提供商实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化DeepSeek客户端"""
        try:
            import openai
            
            api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                self.logger.warning("未找到DeepSeek API密钥")
                return
            
            # DeepSeek使用OpenAI兼容的API
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.base_url or "https://api.deepseek.com",
                timeout=self.config.timeout
            )
            
            self.logger.info("DeepSeek客户端初始化成功")
            
        except ImportError:
            self.logger.error("未安装openai库，请运行: pip install openai")
        except Exception as e:
            self.logger.error(f"初始化DeepSeek客户端失败: {e}")
    
    def chat_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """DeepSeek聊天完成"""
        if not self.client:
            raise RuntimeError("DeepSeek客户端未初始化")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取 token 使用信息
            if hasattr(response, 'usage') and response.usage:
                self.last_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self.logger.debug(f"Token 使用: {self.last_token_usage}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"DeepSeek API调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查DeepSeek是否可用"""
        return self.client is not None


class LLMProviderFactory:
    """LLM提供商工厂类"""
    
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "deepseek": DeepSeekProvider,
        "mock": MockProvider
    }
    
    @classmethod
    def create_provider_from_specialization(cls, specialization: str = "default") -> LLMProvider:
        """
        从 agents 配置的 specialization 创建提供商（推荐使用）
        
        参数:
            specialization: 专门化类型，如 'default', 'test', 'conversation', 'data_extraction' 等
            
        返回:
            LLMProvider: 提供商实例
        """
        if not USE_AGENTS_CONFIG:
            raise RuntimeError(
                "无法使用 agents 配置。请确保 agents/llm_config.yaml 存在并且可访问"
            )
        
        # 加载 agents 的模型配置
        model_config = get_model_config(specialization)
        
        # 转换为 LLMConfig 格式
        base_url = getattr(model_config, 'base_url', None) if hasattr(model_config, 'base_url') else None
        llm_config = LLMConfig(
            model_name=model_config.model,
            api_key=model_config.api_key,
            timeout=model_config.timeout,
            base_url=base_url,  # 支持 base_url
            temperature=0.1,  # 默认值
            max_tokens=4000   # 默认值
        )
        
        # 根据类型创建对应的提供商
        provider_type = model_config.type.lower()
        # 如果类型是 openai 但 base_url 是 deepseek，使用 deepseek provider
        if provider_type == "openai" and base_url and "deepseek" in base_url.lower():
            provider_type = "deepseek"
        elif provider_type not in cls._providers:
            # 如果类型不在映射中，尝试使用 openai（因为很多API兼容OpenAI）
            provider_type = "openai"
        
        return cls._providers[provider_type](llm_config)
    
    @classmethod
    def create_provider(cls, provider_name: str, config: LLMConfig) -> LLMProvider:
        """
        创建LLM提供商实例
        
        参数:
            provider_name: 提供商名称 (openai, anthropic, ollama, mock)
            config: LLM配置
            
        返回:
            LLMProvider: 提供商实例
        """
        if provider_name not in cls._providers:
            raise ValueError(f"不支持的提供商: {provider_name}")
        
        return cls._providers[provider_name](config)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """
        获取所有提供商的可用性状态
        
        返回:
            Dict[str, bool]: 提供商名称到可用性的映射
        """
        availability = {}
        
        for name, provider_class in cls._providers.items():
            try:
                # 创建一个临时配置来测试可用性
                temp_config = LLMConfig(model_name="test")
                provider = provider_class(temp_config)
                availability[name] = provider.is_available()
            except Exception:
                availability[name] = False
        
        return availability
    
    @classmethod
    def auto_select_provider(cls, preferred_providers: list = None) -> str:
        """
        自动选择可用的提供商
        
        参数:
            preferred_providers: 优先选择的提供商列表
            
        返回:
            str: 选择的提供商名称
        """
        if preferred_providers is None:
            preferred_providers = ["openai", "anthropic", "ollama", "mock"]
        
        availability = cls.get_available_providers()
        
        for provider in preferred_providers:
            if provider in availability and availability[provider]:
                return provider
        
        # 如果没有找到可用的提供商，返回mock
        return "mock"
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json
import logging
import re
import time
from ..config import LLMConfig
from ..llm_providers import LLMProviderFactory, LLMProvider
from ..llm_logger import get_llm_logger
from ..token_tracker import get_token_tracker


class BaseAgent(ABC):
    """基础Agent抽象类，定义所有Agent的通用接口"""
    
    def __init__(self, config: LLMConfig, name: str = "BaseAgent", provider_name: str = "mock"):
        self.config = config
        self.name = name
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"SDCCheck.{name}")
        
        # 初始化LLM提供商
        self.llm_provider = LLMProviderFactory.create_provider(provider_name, config)
        
        if not self.llm_provider.is_available():
            self.logger.warning(f"提供商 {provider_name} 不可用，回退到mock提供商")
            self.llm_provider = LLMProviderFactory.create_provider("mock", config)
        
        # Token 追踪器
        self.token_tracker = get_token_tracker()
    

    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理输入数据的抽象方法，子类必须实现"""
        pass
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, 
                  constraint_name: str = "") -> str:
        """
        调用LLM的通用方法
        
        参数:
            prompt: 用户提示
            system_prompt: 系统提示
            constraint_name: 当前处理的约束名称（用于 token 追踪）
        """
        # 获取LLM日志记录器
        llm_logger = get_llm_logger()
        
        # 记录开始时间
        start_time = time.time()
        response = None
        error_msg = None
        
        try:
            self.logger.debug(f"调用LLM，prompt长度: {len(prompt)}")
            
            response = self.llm_provider.chat_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            self.logger.debug(f"LLM响应长度: {len(response)}")
            
            # Token 追踪：获取并累加 token 使用信息
            token_usage = self.llm_provider.get_last_token_usage()
            if token_usage and token_usage.get("total_tokens", 0) > 0:
                self.token_tracker.add_usage(
                    usage=token_usage,
                    model=self.config.model_name,
                    agent_name=self.name,
                    constraint_name=constraint_name
                )
                self.logger.debug(f"Token 追踪 - {self.name}: {token_usage}")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"调用LLM时出错: {e}")
            raise
        
        finally:
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录LLM交互详情
            llm_logger.log_llm_interaction(
                agent_name=self.name,
                provider_name=self.provider_name,
                model_name=self.config.model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                response=response or "",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                execution_time=execution_time,
                error=error_msg
            )
        
        return response
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的JSON响应"""
        try:
            # 尝试提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                
 
                json_str = re.sub(r'\\\s+', ' ', json_str)
                
                return json.loads(json_str)
            else:
 
                cleaned_response = re.sub(r'\\\s+', ' ', response)
                return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"解析JSON响应失败: {e}")
            self.logger.error(f"原始响应: {response}")
            raise ValueError(f"无法解析LLM响应为JSON: {e}")
    
    def _format_data_for_prompt(self, data: Any) -> str:
        """将数据格式化为适合prompt的字符串"""
        if isinstance(data, dict):
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif isinstance(data, list):
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return str(data)
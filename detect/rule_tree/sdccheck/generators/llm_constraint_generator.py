from typing import List, Dict, Any

from ..models import SchemaInfo, TrainingConfig, Constraint
from ..agents import ConstraintAgent
from ..config import LLMConfig


class LLMConstraintGenerator:
    """基于LLM的约束生成器，使用智能Agent生成约束规则"""
    
    def __init__(self, llm_config: LLMConfig = None, provider_name: str = "mock"):
        """
        初始化LLM约束生成器
        
        参数:
            llm_config: LLM配置，如果为None则使用默认配置
            provider_name: LLM提供商名称
        """
        if llm_config is None:
            llm_config = LLMConfig(
                model_name="gpt-4",
                temperature=0.1,
                max_tokens=2000
            )
        
        self.agent = ConstraintAgent(llm_config, provider_name)
        self.provider_name = provider_name
    
    def generate(self, schema_info: SchemaInfo, config: TrainingConfig) -> List[Constraint]:
        """
        根据数据库模式和训练配置生成约束列表
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置，包含DP/TP等并行信息
            
        返回:
            List[Constraint]: 生成的约束列表
        """
        input_data = {
            "schema_info": schema_info,
            "training_config": config
        }
        
        return self.agent.process(input_data)
    
    def update_llm_config(self, llm_config: LLMConfig, provider_name: str = None) -> None:
        """
        更新LLM配置
        
        参数:
            llm_config: 新的LLM配置
            provider_name: 新的提供商名称（可选）
        """
        if provider_name:
            self.provider_name = provider_name
        
        self.agent = ConstraintAgent(llm_config, provider_name or self.provider_name)
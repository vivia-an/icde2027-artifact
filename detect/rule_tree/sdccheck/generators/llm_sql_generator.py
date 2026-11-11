from ..models import Constraint
from ..agents import SQLAgent
from ..config import LLMConfig


class LLMSQLGenerator:
    """基于LLM的SQL生成器，使用智能Agent生成SQL查询语句"""
    
    def __init__(self, llm_config: LLMConfig = None, provider_name: str = "mock"):
        """
        初始化LLM SQL生成器
        
        参数:
            llm_config: LLM配置，如果为None则使用默认配置
            provider_name: LLM提供商名称
        """
        if llm_config is None:
            llm_config = LLMConfig(
                model_name="gpt-4",
                temperature=0.1,
                max_tokens=1500
            )
        
        self.agent = SQLAgent(llm_config, provider_name)
        self.provider_name = provider_name
    
    def generate(self, constraint: Constraint) -> str:
        """
        根据约束对象生成SQL查询语句
        
        参数:
            constraint: 约束对象
            
        返回:
            str: 生成的SQL查询语句
        """
        return self.agent.process(constraint)
    
    def update_llm_config(self, llm_config: LLMConfig, provider_name: str = None) -> None:
        """
        更新LLM配置
        
        参数:
            llm_config: 新的LLM配置
            provider_name: 新的提供商名称（可选）
        """
        if provider_name:
            self.provider_name = provider_name
        
        self.agent = SQLAgent(llm_config, provider_name or self.provider_name)
    
    def validate_sql(self, sql: str) -> bool:
        """
        验证生成的SQL语法是否正确
        
        参数:
            sql: 要验证的SQL语句
            
        返回:
            bool: SQL是否有效
        """
        return self.agent.validate_sql(sql)
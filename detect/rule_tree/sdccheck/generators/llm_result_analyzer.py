import pandas as pd
from typing import Dict, Any

from ..models import Constraint, AnalysisReport
from ..agents import AnalysisAgent
from ..config import LLMConfig


class LLMResultAnalyzer:
    """基于LLM的结果分析器，使用智能Agent分析SQL查询结果"""
    
    def __init__(self, llm_config: LLMConfig = None, provider_name: str = "mock"):
        """
        初始化LLM结果分析器
        
        参数:
            llm_config: LLM配置，如果为None则使用默认配置
            provider_name: LLM提供商名称
        """
        if llm_config is None:
            llm_config = LLMConfig(
                model_name="gpt-4",
                temperature=0.2,  # 分析时可以稍微提高创造性
                max_tokens=1500
            )
        
        self.agent = AnalysisAgent(llm_config, provider_name)
        self.provider_name = provider_name
    
    def analyze(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """
        分析SQL查询结果，判断是否违反约束
        
        参数:
            constraint: 约束对象
            result: SQL查询结果
            
        返回:
            AnalysisReport: 分析报告
        """
        input_data = {
            "constraint": constraint,
            "result": result
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
        
        self.agent = AnalysisAgent(llm_config, self.provider_name)
    
    def batch_analyze(self, constraint_results: list) -> list:
        """
        批量分析多个约束的结果
        
        参数:
            constraint_results: 包含(constraint, result)元组的列表
            
        返回:
            list: AnalysisReport列表
        """
        reports = []
        
        for constraint, result in constraint_results:
            try:
                report = self.analyze(constraint, result)
                reports.append(report)
            except Exception as e:
                # 如果单个分析失败，记录错误但继续处理其他的
                self.agent.logger.error(f"分析约束[{constraint.name}]时出错: {e}")
                # 创建一个错误报告
                error_report = AnalysisReport(
                    constraint=constraint,
                    status="error",
                    summary=f"分析过程中出错: {str(e)}",
                    violations=[],
                    raw_data=result
                )
                reports.append(error_report)
        
        return reports
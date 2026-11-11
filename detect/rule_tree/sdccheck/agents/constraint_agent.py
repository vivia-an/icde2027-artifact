from typing import List, Dict, Any
import json

from .base_agent import BaseAgent, LLMConfig
from ..models import SchemaInfo, TrainingConfig, Constraint, ConstraintType


class ConstraintAgent(BaseAgent):
    """约束生成Agent，使用LLM智能生成约束规则"""
    
    def __init__(self, config: LLMConfig, provider_name: str = "mock"):
        super().__init__(config, "ConstraintAgent", provider_name)
    
    def process(self, input_data: Dict[str, Any]) -> List[Constraint]:
        """处理输入数据并生成约束列表"""
        schema_info = input_data["schema_info"]
        training_config = input_data["training_config"]
        
        return self.generate_constraints(schema_info, training_config)
    
    def generate_constraints(self, schema_info: SchemaInfo, config: TrainingConfig) -> List[Constraint]:
        """
        使用LLM根据数据库模式和训练配置生成约束列表
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置
            
        返回:
            List[Constraint]: 生成的约束列表
        """
        self.logger.info(f"开始生成约束，DP={config.dp}, TP={config.tp}, PP={config.pp}")
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 构建用户提示
        user_prompt = self._build_user_prompt(schema_info, config)
        
        # 调用LLM
        response = self._call_llm(user_prompt, system_prompt)
        
        # 解析响应
        try:
            parsed_response = self._parse_json_response(response)
            constraints = self._convert_to_constraints(parsed_response)
            
            self.logger.info(f"成功生成{len(constraints)}个约束")
            return constraints
            
        except Exception as e:
            self.logger.error(f"解析约束生成结果失败: {e}")
            # 返回空列表或使用fallback策略
            return self._fallback_constraints(schema_info, config)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """
你是一个专业的分布式训练数据一致性检查专家。你的任务是根据数据库模式信息和训练配置，生成合适的约束规则来验证分布式训练检查点的数据一致性。

你需要理解以下概念：
1. 数据并行(DP): 相同的模型在不同设备上处理不同的数据批次，权重应该保持一致
2. 张量并行(TP): 模型的张量被分割到不同设备上，不同设备应该有不同的权重分片
3. 流水线并行(PP): 模型的不同层分布在不同设备上
4. ZeRO优化: 优化器状态、梯度或参数的分片存储

约束类型包括：
- consistency: 一致性约束，检查相同数据在不同设备间是否一致
- partition: 分区约束，检查数据是否正确分片
- completeness: 完整性约束，检查是否包含所有必要的数据
- validity: 有效性约束，检查数据是否有效

请根据输入的数据库模式和训练配置，生成相应的约束规则。
        """
    
    def _build_user_prompt(self, schema_info: SchemaInfo, config: TrainingConfig) -> str:
        """构建用户提示"""
        schema_str = self._format_data_for_prompt(schema_info.dict())
        config_str = self._format_data_for_prompt(config.dict())
        
        return f"""
请根据以下数据库模式信息和训练配置，生成合适的约束规则：

数据库模式信息：
{schema_str}

训练配置：
{config_str}

请分析数据库中的表结构，识别可能包含训练检查点数据的表，然后根据训练配置生成相应的约束规则。

要求：
1. 如果DP > 1，生成数据并行一致性约束
2. 如果TP > 1，生成张量并行分片约束
3. 如果PP > 1，生成流水线并行相关约束
4. 如果zero_stage > 0，生成ZeRO优化相关约束
5. 总是生成模型完整性和有效性约束

请以JSON格式返回结果，格式如下：
{{
    "constraints": [
        {{
            "name": "约束名称",
            "description": "约束描述",
            "type": "约束类型(consistency/partition/completeness/validity)",
            "logic": "约束逻辑表达式",
            "tables": ["涉及的表名"],
            "params": {{"参数名": "参数值"}}
        }}
    ],
    "reasoning": "生成这些约束的原因和逻辑"
}}
        """
    
    def _convert_to_constraints(self, parsed_response: Dict[str, Any]) -> List[Constraint]:
        """将解析后的响应转换为Constraint对象列表"""
        constraints = []
        
        for constraint_data in parsed_response.get("constraints", []):
            try:
                constraint = Constraint(
                    name=constraint_data["name"],
                    description=constraint_data["description"],
                    type=ConstraintType(constraint_data["type"]),
                    logic=constraint_data["logic"],
                    tables=constraint_data["tables"],
                    params=constraint_data.get("params", {})
                )
                constraints.append(constraint)
            except Exception as e:
                self.logger.warning(f"跳过无效的约束定义: {e}")
                continue
        
        return constraints
    
    def _fallback_constraints(self, schema_info: SchemaInfo, config: TrainingConfig) -> List[Constraint]:
        """当LLM生成失败时的fallback策略"""
        self.logger.warning("使用fallback策略生成基础约束")
        
        constraints = []
        
        # 查找可能的检查点表
        checkpoint_tables = self._find_checkpoint_tables(schema_info)
        
        if not checkpoint_tables:
            return constraints
        
        table = checkpoint_tables[0]  # 使用第一个找到的表
        
        # 生成基础约束
        if config.dp > 1:
            constraints.append(Constraint(
                name="DP权重一致性",
                description="数据并行权重一致性检查",
                type=ConstraintType.CONSISTENCY,
                logic="GROUP BY name, tp -> (COUNT(DISTINCT dp) > 1) IMPLIES (COUNT(DISTINCT cksum) = 1)",
                tables=[table],
                params={}
            ))
        
        if config.tp > 1:
            constraints.append(Constraint(
                name="TP权重分片",
                description="张量并行权重分片检查",
                type=ConstraintType.PARTITION,
                logic="GROUP BY name, dp -> (COUNT(DISTINCT tp) > 1) IMPLIES (COUNT(DISTINCT cksum) > 1)",
                tables=[table],
                params={}
            ))
        
        return constraints
    
    def _find_checkpoint_tables(self, schema_info: SchemaInfo) -> List[str]:
        """查找可能包含检查点数据的表"""
        checkpoint_tables = []
        
        for table_name, table_info in schema_info.tables.items():
            # 检查表名是否包含关键字
            if any(keyword in table_name.lower() for keyword in ["checkpoint", "ckpt", "weight", "model", "tensor"]):
                checkpoint_tables.append(table_name)
                continue
                
            # 检查列名是否包含关键字
            columns = [col["name"] for col in table_info.get("columns", [])]
            if any(col in columns for col in ["name", "weight", "tensor", "model", "layer"]):
                checkpoint_tables.append(table_name)
                continue
        
        return checkpoint_tables
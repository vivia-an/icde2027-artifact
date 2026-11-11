from typing import Dict, Any
import pandas as pd
import json

from .base_agent import BaseAgent, LLMConfig
from ..models import Constraint, ConstraintType, AnalysisReport, ResultStatus


class AnalysisAgent(BaseAgent):
    """结果分析Agent，使用LLM智能分析SQL查询结果"""
    
    def __init__(self, config: LLMConfig, provider_name: str = "mock"):
        super().__init__(config, "AnalysisAgent", provider_name)
    
    def process(self, input_data: Dict[str, Any]) -> AnalysisReport:
        """处理输入数据并生成分析报告"""
        constraint = input_data["constraint"]
        result = input_data["result"]
        
        return self.analyze_result(constraint, result)
    
    def analyze_result(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """
        使用LLM分析SQL查询结果并生成分析报告
        
        参数:
            constraint: 约束对象
            result: SQL查询结果
            
        返回:
            AnalysisReport: 分析报告
        """
        self.logger.info(f"开始分析约束[{constraint.name}]的查询结果")
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 构建用户提示
        user_prompt = self._build_user_prompt(constraint, result)
        
        # 调用LLM
        response = self._call_llm(user_prompt, system_prompt)
        
        # 解析响应
        try:
            parsed_response = self._parse_json_response(response)
            report = self._convert_to_analysis_report(constraint, result, parsed_response)
            
            self.logger.info(f"分析完成，状态: {report.status}")
            return report
            
        except Exception as e:
            self.logger.error(f"解析分析结果失败: {e}")
            # 使用fallback策略
            return self._fallback_analysis(constraint, result)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """
你是一个专业的数据分析专家，专门分析分布式训练数据检查的结果。

你需要理解以下约束类型的分析方法：

1. consistency（一致性或者不一致性约束）:
   - 如果查询结果为空，说明没有违规，状态为PASS
   - 如果有结果，说明存在的情况，状态为FAIL
   - 分析违规的具体原因和影响

2. partition（分区约束）:
   - 如果查询结果为空，说明分片正确，状态为PASS
   - 如果有结果，说明分片有问题，状态为FAIL
   - 分析分片错误的类型和位置

3. completeness（完整性约束）:
   - 比较实际数量与期望数量
   - 如果匹配则PASS，否则FAIL
   - 分析缺失或多余的数据

4. validity（有效性约束）:
   - 如果查询结果为空，说明所有数据有效，状态为PASS
   - 如果有结果，说明存在无效数据，状态为FAIL
   - 分析无效数据的类型和原因

分析结果应该包括：
- 状态判断（PASS/FAIL/WARNING/ERROR）
- 简洁的摘要说明
- 违规数据的详细信息
- 修复建议
- 置信度评估
        """
    
    def _build_user_prompt(self, constraint: Constraint, result: pd.DataFrame) -> str:
        """构建用户提示"""
        constraint_str = self._format_data_for_prompt(constraint.dict())
        
        # 处理查询结果
        if result.empty:
            result_str = "查询结果为空（0行数据）"
        else:
            # 限制结果大小以避免prompt过长
            sample_size = min(10, len(result))
            result_sample = result.head(sample_size)
            result_str = f"查询结果（共{len(result)}行，显示前{sample_size}行）：\n{result_sample.to_string()}"
            
            # 添加统计信息
            result_str += f"\n\n统计信息：\n"
            result_str += f"- 总行数: {len(result)}\n"
            result_str += f"- 列数: {len(result.columns)}\n"
            result_str += f"- 列名: {list(result.columns)}\n"
        
        return f"""
请分析以下约束检查的结果：

约束信息：
{constraint_str}

查询结果：
{result_str}

请根据约束类型和查询结果，进行以下分析：

1. 判断约束检查的状态（PASS/FAIL/WARNING/ERROR）
2. 提供简洁明确的摘要说明
3. 如果有违规，列出具体的违规情况
4. 提供修复建议
5. 评估分析的置信度

分析原则：
- 对于检测违规的查询：结果为空表示PASS，有结果表示FAIL
- 对于统计类查询：需要比较实际值与期望值
- 考虑数据的业务含义和分布式训练的特点
- 提供具体可行的修复建议

请以JSON格式返回分析结果：
{{
    "status": "状态(pass/fail/warning/error)",
    "summary": "简洁的摘要说明",
    "violations": [
        {{"字段名": "违规值", "描述": "违规描述"}}
    ],
    "suggestion": "修复建议",
    "confidence": "置信度(0.0-1.0)",
    "reasoning": "分析推理过程"
}}
        """
    
    def _convert_to_analysis_report(self, constraint: Constraint, result: pd.DataFrame, 
                                   parsed_response: Dict[str, Any]) -> AnalysisReport:
        """将解析后的响应转换为AnalysisReport对象"""
        try:
            status_str = parsed_response.get("status", "error").lower()
            status = ResultStatus(status_str)
        except ValueError:
            status = ResultStatus.ERROR
        
        return AnalysisReport(
            constraint=constraint,
            status=status,
            summary=parsed_response.get("summary", "分析完成"),
            violations=parsed_response.get("violations", []),
            suggestion=parsed_response.get("suggestion"),
            raw_data=result
        )
    
    def _fallback_analysis(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """当LLM分析失败时的fallback策略"""
        self.logger.warning(f"使用fallback策略分析约束[{constraint.name}]")
        
        # 基于规则的简单分析
        if result.empty:
            # 查询结果为空，通常表示没有违规
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.PASS,
                summary=f"通过: 未发现违反{constraint.name}约束的情况",
                violations=[],
                raw_data=result
            )
        else:
            # 有查询结果，根据约束类型进行分析
            if constraint.type == ConstraintType.CONSISTENCY:
                return self._analyze_consistency_fallback(constraint, result)
            elif constraint.type == ConstraintType.PARTITION:
                return self._analyze_partition_fallback(constraint, result)
            elif constraint.type == ConstraintType.COMPLETENESS:
                return self._analyze_completeness_fallback(constraint, result)
            elif constraint.type == ConstraintType.VALIDITY:
                return self._analyze_validity_fallback(constraint, result)
            else:
                return AnalysisReport(
                    constraint=constraint,
                    status=ResultStatus.ERROR,
                    summary=f"错误: 不支持的约束类型 {constraint.type}",
                    violations=[],
                    raw_data=result
                )
    
    def _analyze_consistency_fallback(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """一致性约束的fallback分析"""
        violations = result.to_dict("records")[:10]
        
        if "DP权重一致性" in constraint.name:
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.FAIL,
                summary=f"失败: 发现{len(result)}个层的权重在不同DP-rank之间不一致",
                violations=violations,
                suggestion="检查这些层在不同DP-rank上的权重是否被正确同步",
                raw_data=result
            )
        else:
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.FAIL,
                summary=f"失败: 发现{len(result)}个记录违反一致性约束",
                violations=violations,
                suggestion="检查违规记录中的数据不一致问题",
                raw_data=result
            )
    
    def _analyze_partition_fallback(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """分区约束的fallback分析"""
        violations = result.to_dict("records")[:10]
        
        if "TP权重分片" in constraint.name:
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.FAIL,
                summary=f"失败: 发现{len(result)}个层在不同TP-rank之间权重相同，应当为不同分片",
                violations=violations,
                suggestion="检查张量并行是否正确实现，不同TP-rank应有不同的权重分片",
                raw_data=result
            )
        else:
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.FAIL,
                summary=f"失败: 发现{len(result)}个记录违反分区约束",
                violations=violations,
                suggestion="检查分区策略是否正确应用",
                raw_data=result
            )
    
    def _analyze_completeness_fallback(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """完整性约束的fallback分析"""
        expected_count = constraint.params.get("expected_layers_count", -1)
        
        if "distinct_names" in result.columns:
            actual_count = result["distinct_names"].iloc[0]
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.WARNING,
                summary=f"警告: 模型包含{actual_count}个不同的层或权重，未指定预期数量",
                violations=[{"actual_count": actual_count}],
                suggestion="请在约束参数中设置expected_layers_count",
                raw_data=result
            )
        elif "actual_count" in result.columns:
            actual_count = result["actual_count"].iloc[0]
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.FAIL,
                summary=f"失败: 模型层数不符合预期，预期{expected_count}个，实际{actual_count}个",
                violations=[{"expected_count": expected_count, "actual_count": actual_count}],
                suggestion="检查模型定义是否完整，以及检查点是否包含所有必要的权重",
                raw_data=result
            )
        else:
            return AnalysisReport(
                constraint=constraint,
                status=ResultStatus.ERROR,
                summary="错误: 无法解析完整性检查结果",
                violations=[],
                raw_data=result
            )
    
    def _analyze_validity_fallback(self, constraint: Constraint, result: pd.DataFrame) -> AnalysisReport:
        """有效性约束的fallback分析"""
        violations = result.to_dict("records")[:10]
        
        return AnalysisReport(
            constraint=constraint,
            status=ResultStatus.FAIL,
            summary=f"失败: 发现{len(result)}个无效或异常的检查点文件",
            violations=violations,
            suggestion="检查这些文件的完整性和有效性，可能需要重新生成检查点",
            raw_data=result
        )
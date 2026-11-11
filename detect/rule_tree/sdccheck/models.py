from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class TrainingConfig(BaseModel):
    """训练配置，包含分布式训练的各种并行配置信息"""
    dp: int = Field(default=1, description="数据并行度")
    tp: int = Field(default=1, description="张量并行度")
    pp: int = Field(default=1, description="流水线并行度")
    ep: int = Field(default=1, description="专家并行度")
    sp: int = Field(default=1, description="序列并行度")
    zero_stage: int = Field(default=0, description="ZeRO优化阶段")


class SchemaInfo(BaseModel):
    """数据库模式信息，描述数据库中的表结构和JSON字段键"""
    tables: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="表信息，键为表名，值为表结构")


class ConstraintType(str, Enum):
    """约束类型枚举"""
    CONSISTENCY = "consistency"  # 一致性约束
    PARTITION = "partition"      # 分区约束
    COMPLETENESS = "completeness"  # 完整性约束
    VALIDITY = "validity"        # 有效性约束


class Constraint(BaseModel):
    """约束对象，描述一个需要验证的业务规则"""
    name: str = Field(..., description="约束名称")
    description: str = Field(..., description="约束描述")
    type: ConstraintType = Field(..., description="约束类型")
    logic: str = Field(..., description="约束逻辑表达式")
    tables: List[str] = Field(..., description="涉及的表")
    params: Dict[str, Any] = Field(default_factory=dict, description="约束参数")
    applicable_conditions: Dict[str, str] = Field(default_factory=dict, description="适用条件")
    children: List[str] = Field(default_factory=list, description="子约束名称列表")
    parent: Optional[str] = Field(None, description="父约束名称")
    level: int = Field(default=0, description="树层级，0表示根节点")
    category: Optional[str] = Field(None, description="约束所属类别")


class ResultStatus(str, Enum):
    """结果状态枚举"""
    PASS = "pass"  # 通过
    FAIL = "fail"  # 失败
    ERROR = "error"  # 错误
    WARNING = "warning"  # 警告


class AnalysisReport(BaseModel):
    """分析报告，包含约束验证结果和详细信息"""
    constraint: Constraint = Field(..., description="原始约束")
    status: ResultStatus = Field(..., description="验证状态")
    summary: str = Field(..., description="结果摘要")
    violations: List[Dict[str, Any]] = Field(default_factory=list, description="违规数据样例")
    suggestion: Optional[str] = Field(None, description="修复建议")
    raw_data: Optional[Any] = Field(None, description="原始查询结果")
    execution_time: Optional[float] = Field(None, description="执行时间（秒）")
    node_id: Optional[str] = Field(None, description="约束树节点ID")


class ConstraintNode(BaseModel):
    """约束树节点"""
    node_id: str = Field(..., description="节点唯一标识")
    constraint: Constraint = Field(..., description="约束对象")
    children: List['ConstraintNode'] = Field(default_factory=list, description="子节点列表")
    parent_id: Optional[str] = Field(None, description="父节点ID")
    level: int = Field(default=0, description="树层级，0表示根节点")
    report: Optional[AnalysisReport] = Field(None, description="约束检查报告")
    
    class Config:
        # 允许自引用
        arbitrary_types_allowed = True


class ConstraintTree(BaseModel):
    """约束树，组织所有约束的层级关系"""
    root_nodes: List[ConstraintNode] = Field(default_factory=list, description="根节点列表")
    node_map: Dict[str, ConstraintNode] = Field(default_factory=dict, description="节点ID到节点的映射")
    max_depth: int = Field(default=0, description="树的最大深度")
    total_nodes: int = Field(default=0, description="总节点数")
    
    class Config:
        arbitrary_types_allowed = True


class FailureChain(BaseModel):
    """失败链路，从根节点到叶子节点的失败路径"""
    path: List[str] = Field(default_factory=list, description="失败节点ID路径")
    root_cause: str = Field(..., description="根本原因（最上层失败节点）")
    affected_nodes: List[str] = Field(default_factory=list, description="受影响的子节点")
    severity: str = Field(..., description="严重程度：critical/high/medium/low")


class RootCauseAnalysis(BaseModel):
    """根因分析结果"""
    analysis_id: str = Field(..., description="分析ID")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="分析时间")
    total_constraints: int = Field(..., description="总约束数")
    passed_constraints: int = Field(..., description="通过的约束数")
    failed_constraints: int = Field(..., description="失败的约束数")
    failure_chains: List[FailureChain] = Field(default_factory=list, description="失败链路列表")
    root_causes: List[Dict[str, Any]] = Field(default_factory=list, description="根本原因列表")
    recommendations: List[str] = Field(default_factory=list, description="修复建议列表")
    constraint_tree: Optional[Dict[str, Any]] = Field(None, description="约束树结构（简化版）")
    execution_summary: Dict[str, Any] = Field(default_factory=dict, description="执行摘要")


class BatchExecutionResult(BaseModel):
    """批量执行结果"""
    batch_id: int = Field(..., description="批次ID")
    node_ids: List[str] = Field(..., description="本批次执行的节点ID列表")
    reports: List[AnalysisReport] = Field(default_factory=list, description="分析报告列表")
    execution_time: float = Field(..., description="批次执行时间（秒）")
    success_count: int = Field(default=0, description="成功执行的数量")
    failure_count: int = Field(default=0, description="失败的数量")
    error_count: int = Field(default=0, description="错误的数量") 
"""根因分析器 - 分析失败约束的根本原因和定位链路

核心逻辑：
- 父节点失败 = 发现该类问题存在（大范围检查）
- 子节点失败 = 真正的根因（小范围，具体定位）
- 失败链路：父节点 → 子节点（从发现问题到定位具体原因）
"""

import json
import logging
import numpy as np  # 添加这行
from typing import List, Dict, Any, Set
from datetime import datetime
from .models import RootCauseAnalysis, FailureChain, ConstraintNode
from .constraint_tree import ConstraintTreeManager


class RootCauseAnalyzer:
    """根因分析器 - 子节点才是真正的根因"""
    
    def __init__(self):
        self.logger = logging.getLogger("SDCCheck.RootCauseAnalyzer")
    
    def analyze(self, tree_manager: ConstraintTreeManager) -> RootCauseAnalysis:
        """分析约束树，找出根本原因（子节点）"""
        if not tree_manager.tree:
            raise ValueError("约束树为空")
        
        self.logger.info("开始根因分析")
        
        # 收集所有失败节点
        failed_nodes = tree_manager.get_failed_nodes()
        
        # 计算统计信息
        total_nodes = tree_manager.tree.total_nodes
        passed_count = sum(1 for node in tree_manager.tree.node_map.values() 
                          if node.report and node.report.status == "pass")
        failed_count = len(failed_nodes)
        
        # 识别失败链路（从父到子的定位链）
        failure_chains = self._identify_failure_chains(tree_manager, failed_nodes)
        
        # 识别根本原因（叶子节点才是真正的根因）
        root_causes = self._identify_root_causes(tree_manager, failed_nodes)
        
        # 生成修复建议
        recommendations = self._generate_recommendations(root_causes, failure_chains)
        
        # 生成树结构
        constraint_tree_dict = tree_manager.to_dict()
        
        # 创建执行摘要
        execution_summary = self._create_execution_summary(tree_manager)
        
        analysis = RootCauseAnalysis(
            analysis_id=f"RCA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_constraints=total_nodes,
            passed_constraints=passed_count,
            failed_constraints=failed_count,
            failure_chains=failure_chains,
            root_causes=root_causes,
            recommendations=recommendations,
            constraint_tree=constraint_tree_dict,
            execution_summary=execution_summary
        )
        
        self.logger.info(
            f"根因分析完成: 发现 {len(root_causes)} 个根本原因（子约束）, "
            f"{len(failure_chains)} 条定位链路"
        )
        
        return analysis
    
    def _identify_failure_chains(self, tree_manager: ConstraintTreeManager, 
                                 failed_nodes: List[ConstraintNode]) -> List[FailureChain]:
        """
        识别失败链路（从父节点到子节点的定位链）
        
        链路描述：父约束发现问题 → 子约束定位具体原因
        """
        chains = []
        processed_leaves: Set[str] = set()
        
        # 找出所有失败的叶子节点（没有子节点或子节点都通过的失败节点）
        leaf_failures = []
        for node in failed_nodes:
            if not node.children:
                # 没有子节点，是叶子
                leaf_failures.append(node)
            else:
                # 检查是否所有子节点都通过
                has_failed_child = any(
                    child.report and child.report.status == "fail" 
                    for child in node.children
                )
                if not has_failed_child:
                    # 子节点都通过，这个节点本身就是问题定位点
                    leaf_failures.append(node)
        
        for leaf_node in leaf_failures:
            if leaf_node.node_id in processed_leaves:
                continue
            
            # 从叶子向上追溯到根，构建定位链（从父到子展示）
            path_to_root = tree_manager.get_path_to_root(leaf_node.node_id)
            
            # 找出路径中所有失败的节点
            failed_path = [node for node in path_to_root if node.report and node.report.status == "fail"]
            
            if not failed_path:
                continue
            
            # 链路：从父（发现问题）到子（定位原因）
            # path_to_root 是从根到叶子的顺序
            chain = FailureChain(
                path=[node.node_id for node in failed_path],  # 从父到子
                root_cause=leaf_node.constraint.name,  # 真正的根因是子节点
                affected_nodes=[node.node_id for node in failed_path if node.node_id != leaf_node.node_id],
                severity=self._calculate_severity(leaf_node, failed_path)
            )
            
            chains.append(chain)
            processed_leaves.add(leaf_node.node_id)
        
        return chains
    
    def _identify_root_causes(self, tree_manager: ConstraintTreeManager,
                             failed_nodes: List[ConstraintNode]) -> List[Dict[str, Any]]:
        """
        识别根本原因（子节点/叶子节点才是真正的根因）
        
        逻辑：
        - 找失败的叶子节点（没有子节点，或子节点都通过）
        - 这些才是真正的具体问题点
        """
        root_causes = []
        identified: Set[str] = set()
        
        for failed_node in failed_nodes:
            # 判断是否是"真正的根因"（叶子节点或子节点都通过）
            is_leaf = not failed_node.children
            children_all_pass = all(
                child.report and child.report.status == "pass"
                for child in failed_node.children
            ) if failed_node.children else True
            
            if not (is_leaf or children_all_pass):
                # 不是叶子，且有子节点失败，跳过（子节点才是根因）
                continue
            
            if failed_node.node_id in identified:
                continue
            
            identified.add(failed_node.node_id)
            
            # 获取从根到此节点的路径（用于展示定位链）
            path_to_root = tree_manager.get_path_to_root(failed_node.node_id)
            parent_chain = []
            for node in path_to_root:
                if node.node_id != failed_node.node_id:
                    parent_chain.append({
                        "node_id": node.node_id,
                        "name": node.constraint.name,
                        "level": node.level,
                        "status": node.report.status if node.report else "unknown"
                    })
            
            root_cause = {
                "node_id": failed_node.node_id,
                "constraint_name": failed_node.constraint.name,
                "description": failed_node.constraint.description,
                "level": failed_node.level,
                "status": failed_node.report.status,
                "summary": failed_node.report.summary,
                "is_leaf": is_leaf,
                # 定位链：从父约束到此约束的路径
                "localization_chain": parent_chain,
                "localization_chain_desc": " → ".join([p["name"] for p in parent_chain] + [failed_node.constraint.name]),
                "violation_examples": failed_node.report.violations[:5] if failed_node.report.violations else [],
                "suggestion": failed_node.report.suggestion
            }
            
            root_causes.append(root_cause)
        
        # 按层级排序（子节点在前，因为是真正的根因）
        root_causes.sort(key=lambda x: -x["level"])
        
        return root_causes
    
    def _calculate_severity(self, leaf_node: ConstraintNode, 
                           path: List[ConstraintNode]) -> str:
        """计算严重程度（基于叶子节点和路径长度）"""
        name_lower = leaf_node.constraint.name.lower()
        
        critical_keywords = ["cksum", "checksum", "梯度", "grad", "optimizer"]
        high_keywords = ["shape", "dtype", "requires_grad", "同步"]
        
        if any(kw in name_lower for kw in critical_keywords):
            return "critical"
        elif any(kw in name_lower for kw in high_keywords):
            return "high"
        elif len(path) > 2:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, root_causes: List[Dict[str, Any]], 
                                 failure_chains: List[FailureChain]) -> List[str]:
        """生成修复建议（基于真正的根因-子节点）"""
        recommendations = []
        
        for root_cause in root_causes:
            constraint_name = root_cause["constraint_name"].lower()
            chain_desc = root_cause.get("localization_chain_desc", "")
            
            if "cksum" in constraint_name or "checksum" in constraint_name:
                recommendations.append(
                    f"【参数校验和不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查对应阶段的参数同步逻辑，可能是 AllReduce 通信问题"
                )
            elif "dtype" in constraint_name:
                recommendations.append(
                    f"【数据类型不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查混合精度配置，确保各 rank 的 dtype 一致"
                )
            elif "shape" in constraint_name:
                recommendations.append(
                    f"【形状不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查模型初始化，确保参数 shape 一致"
                )
            elif "requires_grad" in constraint_name:
                recommendations.append(
                    f"【requires_grad不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查参数冻结逻辑，确保各 rank 的 requires_grad 设置一致"
                )
            elif "grad" in constraint_name or "梯度" in constraint_name:
                recommendations.append(
                    f"【梯度不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查 backward 阶段的梯度同步，可能是 AllReduce 问题"
                )
            elif "optimizer" in constraint_name:
                recommendations.append(
                    f"【优化器状态不一致】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   建议: 检查优化器初始化和状态同步逻辑"
                )
            else:
                recommendations.append(
                    f"【一致性检查失败】{root_cause['constraint_name']}\n"
                    f"   定位链: {chain_desc}\n"
                    f"   摘要: {root_cause['summary']}"
                )
            
            if root_cause.get("suggestion"):
                recommendations.append(f"   原始建议: {root_cause['suggestion']}")
        
        # 去重
        recommendations = list(dict.fromkeys(recommendations))
        
        return recommendations
    
    def _create_execution_summary(self, tree_manager: ConstraintTreeManager) -> Dict[str, Any]:
        """创建执行摘要"""
        if not tree_manager.tree:
            return {}
        
        level_stats = {}
        for level in range(tree_manager.tree.max_depth + 1):
            nodes = tree_manager.get_nodes_by_level(level)
            
            passed = sum(1 for n in nodes if n.report and n.report.status == "pass")
            failed = sum(1 for n in nodes if n.report and n.report.status == "fail")
            errors = sum(1 for n in nodes if n.report and n.report.status == "error")
            unchecked = sum(1 for n in nodes if n.report is None)
            
            level_name = "父约束" if level == 0 else "子约束"
            level_stats[f"level_{level}_{level_name}"] = {
                "total": len(nodes),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "unchecked": unchecked
            }
        
        total_time = 0.0
        for node in tree_manager.tree.node_map.values():
            if node.report and node.report.execution_time:
                total_time += node.report.execution_time
        
        return {
            "max_depth": tree_manager.tree.max_depth,
            "total_nodes": tree_manager.tree.total_nodes,
            "level_statistics": level_stats,
            "total_execution_time": round(total_time, 2)
        }
    
    def export_to_json(self, analysis: RootCauseAnalysis, output_path: str) -> None:
        """导出分析结果到JSON文件"""
        try:
            # 自定义 JSON encoder 处理 ndarray
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    return super().default(obj)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.dict(), f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            self.logger.info(f"根因分析结果已导出到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"导出JSON失败: {e}")
            raise
    
    def print_summary(self, analysis: RootCauseAnalysis) -> str:
        """生成分析结果的文本摘要"""
        lines = []
        lines.append("=" * 80)
        lines.append("根因分析报告")
        lines.append("=" * 80)
        lines.append(f"分析ID: {analysis.analysis_id}")
        lines.append(f"时间: {analysis.timestamp}")
        lines.append("")
        
        # 总体统计
        lines.append("【总体统计】")
        lines.append(f"  总约束数: {analysis.total_constraints}")
        lines.append(f"  通过: {analysis.passed_constraints}")
        lines.append(f"  失败: {analysis.failed_constraints}")
        if analysis.total_constraints > 0:
            lines.append(f"  通过率: {analysis.passed_constraints / analysis.total_constraints * 100:.1f}%")
        lines.append("")
        
        # 根本原因（子节点）
        if analysis.root_causes:
            lines.append(f"【根本原因】发现 {len(analysis.root_causes)} 个具体问题:")
            lines.append("-" * 80)
            for i, root_cause in enumerate(analysis.root_causes, 1):
                lines.append(f"\n{i}. 【根因】{root_cause['constraint_name']}")
                lines.append(f"   层级: Level {root_cause['level']} ({'叶子节点' if root_cause.get('is_leaf') else '中间节点'})")
                
                # 显示定位链
                if root_cause.get('localization_chain_desc'):
                    lines.append(f"   定位链: {root_cause['localization_chain_desc']}")
                
                lines.append(f"   状态: {root_cause['status']}")
                lines.append(f"   摘要: {root_cause['summary']}")
                
                # 显示违规示例
                if root_cause.get('violation_examples'):
                    lines.append(f"   违规示例:")
                    for j, v in enumerate(root_cause['violation_examples'][:3], 1):
                        lines.append(f"     {j}. {v}")
        
        lines.append("")
        
        # 失败链路
        if analysis.failure_chains:
            lines.append(f"【定位链路】{len(analysis.failure_chains)} 条:")
            lines.append("-" * 80)
            for i, chain in enumerate(analysis.failure_chains, 1):
                lines.append(f"\n链路 {i}:")
                lines.append(f"  根因(子约束): {chain.root_cause}")
                lines.append(f"  严重程度: {chain.severity}")
                lines.append(f"  父约束数: {len(chain.affected_nodes)}")
        
        lines.append("")
        
        # 修复建议
        if analysis.recommendations:
            lines.append("【修复建议】")
            lines.append("-" * 80)
            for i, rec in enumerate(analysis.recommendations, 1):
                lines.append(f"\n{i}. {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

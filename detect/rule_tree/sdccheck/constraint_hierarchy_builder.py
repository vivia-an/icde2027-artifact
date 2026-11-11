"""约束层级构建器 - 基于约束聚类生成两层树结构

核心设计：
1. 只做一次聚类，生成两层树
2. Level 0: 聚类后的公共规则（父约束）
3. Level 1: 原始约束（叶子节点）
4. 父约束的 description 是子约束 description 的公共部分
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from .models import Constraint, ConstraintType
from .config import LLMConfig
from .llm_providers import LLMProviderFactory


class ConstraintHierarchyBuilder:
    """基于约束聚类构建两层层级结构
    
    只做一次聚类：
    - Level 0: 父约束（聚类的公共规则）
    - Level 1: 子约束（原始约束）
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, provider_name: str = "openai"):
        self.logger = logging.getLogger("SDCCheck.ConstraintHierarchyBuilder")
        
        if llm_config is None:
            llm_config = LLMConfig(
                model_name="gpt-4",
                temperature=0.3,
                max_tokens=8000
            )
        
        self.llm_config = llm_config
        self.provider = LLMProviderFactory.create_provider(provider_name, llm_config)
        self.logger.info(f"初始化约束层级构建器（两层聚类模式）")
    
    def build_hierarchy(self, constraints: List[Constraint], target_depth: int = 2) -> List[Constraint]:
        """
        构建两层约束层级结构
        
        参数:
            constraints: 原始约束列表
            target_depth: 忽略，固定为2层
            
        返回:
            List[Constraint]: 两层层级结构的约束列表
        """
        self.logger.info(f"开始构建两层约束层级，共{len(constraints)}个原始约束")
        
        # 提取每个约束的聚类键（stage + 检查目标）
        constraint_with_key = []
        for c in constraints:
            stage = self._extract_stage(c)
            target = self._extract_check_target(c)
            cluster_key = f"{stage}|{target}"
            constraint_with_key.append({
                "constraint": c,
                "stage": stage,
                "target": target,
                "cluster_key": cluster_key
            })
        
        # 按聚类键分组
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for item in constraint_with_key:
            clusters[item["cluster_key"]].append(item)
        
        all_constraints = []
        
        # 收集所有子约束名称（用于避免父约束名称冲突）
        all_child_names = set(c.name for c in constraints)
        
        # 为每个聚类创建父约束
        for cluster_key, items in clusters.items():
            stage, target = cluster_key.split("|", 1)  # 修复：只分割一次
            child_constraints = [item["constraint"] for item in items]
            
            # ============ Level 0: 父约束（公共规则） ============
            # 从子约束名称的公共前缀提取父约束名称
            child_names = [c.name for c in child_constraints]
            parent_name = self._extract_common_name(child_names, all_child_names)
            
            # 从子约束描述的公共前缀提取父约束描述
            parent_desc = self._extract_common_description(child_constraints)
            parent_conditions = self._merge_conditions(child_constraints)
            
            parent_constraint = Constraint(
                name=parent_name,
                description=parent_desc,
                type=ConstraintType.CONSISTENCY,
                logic="",
                tables=["coredump"],
                params={},
                applicable_conditions=parent_conditions
            )
            parent_constraint.level = 0
            parent_constraint.parent = None
            parent_constraint.children = []
            all_constraints.append(parent_constraint)
            
            # ============ Level 1: 子约束（原始约束） ============
            for child in child_constraints:
                child.level = 1
                child.parent = parent_name
                child.children = []
                all_constraints.append(child)
                parent_constraint.children.append(child.name)
        
        self.logger.info(f"两层层级构建完成: {len(clusters)}个父约束, {len(constraints)}个子约束")
        
        return all_constraints
    
    def _extract_stage(self, constraint: Constraint) -> str:
        """提取 stage"""
        name = constraint.name
        patterns = [
            (r'model-before-backward', 'model-before-backward'),
            (r'model-after-backward', 'model-after-backward'),
            (r'model-before-optimizer-step', 'model-before-optimizer-step'),
            (r'model-after-optimizer-step', 'model-after-optimizer-step'),
            (r'main-grad-in-backward', 'main-grad-in-backward'),
        ]
        
        for pattern, stage in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return stage
        
        if constraint.applicable_conditions:
            stage_cond = constraint.applicable_conditions.get('stage', '')
            if stage_cond:
                match = re.search(r"'([^']+)'", stage_cond)
                if match:
                    return match.group(1)
        
        return "general"
    
    def _extract_check_target(self, constraint: Constraint) -> str:
        """提取检查目标"""
        name = constraint.name.lower()
        desc = (constraint.description or "").lower()
        
        targets = [
            ("optimizer_state_dict", ["optimizer_state_dict"]),
            ("optimizer_state", ["optimizer_state", "优化器状态"]),
            ("main_grad", ["main_grad", "主梯度", "main-grad"]),
            ("grad", ["grad_cksum", "梯度cksum", "梯度校验"]),
            ("cksum", ["参数cksum", "cksum一致性", "校验和", "cksum"]),
            ("requires_grad", ["requires_grad"]),
            ("dtype", ["dtype", "数据类型"]),
            ("shape", ["shape", "形状"]),
            ("分布统计", ["分布", "统计", "均值", "方差"]),
            ("bitwise", ["bitwise", "位级"]),
            ("参数", ["参数一致", "参数属性"]),
        ]
        
        for target, keywords in targets:
            for kw in keywords:
                if kw in name or kw in desc:
                    return target
        
        return "general"
    
    
    def _extract_common_name(self, names: List[str], existing_names: Set[str]) -> str:
        """从子约束名称列表中提取公共前缀，生成父约束名称
        
        确保生成的名称不与现有子约束名称冲突
        """
        if not names:
            return "一致性检查"
        
        if len(names) == 1:
            # 单个约束，在名称前加前缀表示是汇总
            return f"[汇总]{names[0]}"
        
        # 找最长公共前缀
        prefix = self._longest_common_prefix(names)
        prefix = prefix.rstrip()
        
        # 确保前缀有意义（至少5个字符）
        if len(prefix) < 5:
            # 使用第一个名称的 stage 部分
            first_name = names[0]
            stage_match = re.search(r'^(model-[a-z-]+阶段)', first_name)
            if stage_match:
                prefix = stage_match.group(1)
            else:
                # 尝试提取 DP/TP 等前缀
                dp_match = re.search(r'^(DP|TP|PP)', first_name)
                if dp_match:
                    prefix = dp_match.group(1)
                else:
                    prefix = "通用"
        
        # 清理前缀末尾不完整的词
        # 如果前缀不是以完整关键词结尾，向前截断到最后一个完整词
        keywords = ['阶段DP', '阶段TP', '阶段PP', '阶段', 'DP', 'TP', 'PP', '参数', '梯度', '状态', '一致性', '检查']
        for kw in keywords:
            if prefix.endswith(kw):
                break
        else:
            # 没有以关键词结尾，尝试截断到最后一个关键词
            for kw in keywords:
                idx = prefix.rfind(kw)
                if idx > 0:
                    prefix = prefix[:idx + len(kw)]
                    break
        
        # 生成父约束名称
        if prefix.endswith('检查'):
            parent_name = prefix
        elif prefix.endswith('一致性'):
            parent_name = prefix + '检查'
        else:
            parent_name = prefix + '一致性检查'
        
        # 确保名称不与子约束冲突
        if parent_name in existing_names:
            parent_name = f"[汇总]{parent_name}"
        
        # 再次检查冲突
        counter = 1
        original_name = parent_name
        while parent_name in existing_names:
            parent_name = f"{original_name}_{counter}"
            counter += 1
        
        return parent_name
    
    def _extract_common_description(self, constraints: List[Constraint]) -> str:
        """从子约束提取公共描述"""
        if not constraints:
            return ""
        
        if len(constraints) == 1:
            return constraints[0].description
        
        descriptions = [c.description for c in constraints if c.description]
        if not descriptions:
            return constraints[0].name
        
        # 找公共前缀
        common = self._longest_common_prefix(descriptions)
        if len(common) > 20:
            return common.rstrip("的，。,. ")
        
        # 提取公共模式
        stage_match = None
        parallel_type = None
        
        for desc in descriptions:
            m = re.search(r'在(model-[a-z-]+)阶段', desc)
            if m:
                stage_match = m.group(1)
            
            if "data parallel" in desc.lower() or "dp组" in desc.lower() or "数据并行" in desc:
                parallel_type = "DP组内"
            elif "tensor parallel" in desc.lower() or "tp组" in desc.lower():
                parallel_type = "TP组内"
        
        if stage_match and parallel_type:
            return f"在{stage_match}阶段，检查{parallel_type}一致性"
        elif stage_match:
            return f"在{stage_match}阶段，检查一致性"
        
        # 返回第一个描述的简化版
        return descriptions[0][:80] if len(descriptions[0]) > 80 else descriptions[0]
    
    def _longest_common_prefix(self, strings: List[str]) -> str:
        """最长公共前缀"""
        if not strings:
            return ""
        shortest = min(strings, key=len)
        for i, char in enumerate(shortest):
            for s in strings:
                if s[i] != char:
                    return shortest[:i]
        return shortest
    
    def _merge_conditions(self, constraints: List[Constraint]) -> Dict[str, str]:
        """合并 applicable_conditions"""
        if not constraints:
            return {}
        
        all_conds: Dict[str, Set[str]] = defaultdict(set)
        for c in constraints:
            if c.applicable_conditions:
                for k, v in c.applicable_conditions.items():
                    all_conds[k].add(v)
        
        merged = {}
        for k, values in all_conds.items():
            if len(values) == 1:
                merged[k] = list(values)[0]
            elif k in ['dp', 'tp', 'pp']:
                merged[k] = "> 1"
        
        return merged
    
    def validate_hierarchy(self, constraints: List[Constraint]) -> bool:
        """验证层级结构"""
        try:
            name_map = {c.name: c for c in constraints}
            
            for c in constraints:
                for child in c.children:
                    if child not in name_map:
                        self.logger.error(f"子节点 '{child}' 不存在")
                        return False
                if c.parent and c.parent not in name_map:
                    self.logger.error(f"父节点 '{c.parent}' 不存在")
                    return False
            
            max_level = max(c.level for c in constraints) if constraints else 0
            root_count = sum(1 for c in constraints if c.parent is None)
            
            self.logger.info(f"层级验证通过: {root_count}个根节点, 深度{max_level}, 总{len(constraints)}节点")
            return True
            
        except Exception as e:
            self.logger.error(f"验证失败: {e}")
            return False

"""预定义约束模块，提供常用的分布式训练数据一致性约束"""

import json
import logging
import os
from typing import List, Dict, Any
from .models import Constraint, ConstraintType, TrainingConfig


class PredefinedConstraints:
    """预定义约束管理器，存储和管理常用的约束规则"""
    
    def __init__(self, config_path: str = None):
        """
        初始化预定义约束
        
        参数:
            config_path: 约束配置文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger(f"SDCCheck.{self.__class__.__name__}")
        self._constraints = self._load_predefined_constraints(config_path)
    
    def _load_predefined_constraints(self, config_path: str = None) -> Dict[str, Dict[str, Constraint]]:
        """从JSON文件加载预定义约束，按类别组织"""
        if config_path:
            constraints_file = config_path
        elif os.environ.get("SDC_CONSTRAINTS_FILE"):
            constraints_file = os.environ["SDC_CONSTRAINTS_FILE"]
            self.logger.info(f"Using constraints file from SDC_CONSTRAINTS_FILE: {constraints_file}")
        else:
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(current_dir)
            constraints_file = os.path.join(project_root, 'config', 'predefined_constraints.json')
        
        if not os.path.exists(constraints_file):
            self.logger.warning(f"约束配置文件不存在: {constraints_file}")
            return self._get_fallback_constraints()
        
        try:
            with open(constraints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            constraints = {}
            constraint_data = data.get('constraints', {})
            
            for category, constraint_dict in constraint_data.items():
                constraints[category] = {}
                for constraint_name, constraint_info in constraint_dict.items():
                    # 获取适用条件
                    applicable_conditions = constraint_info.get('applicable_conditions', {})
                    
                    # 如果有_skip标记，跳过该约束
                    if applicable_conditions.get('_skip') == "= true":
                        self.logger.debug(f"跳过约束 '{constraint_name}' (标记为_skip)")
                        continue
                        
                    constraint = Constraint(
                        name=constraint_info['name'],
                        description=constraint_info['description'],
                        type=ConstraintType(constraint_info['type']),
                        logic=constraint_info['logic'],
                        tables=constraint_info['tables'],
                        params=constraint_info.get('params', {}),
                        applicable_conditions=applicable_conditions
                    )
                    constraints[category][constraint_name] = constraint
            
            self.logger.info(f"成功从JSON文件加载了 {len(constraints)} 个约束类别")
            return constraints
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"加载约束配置文件失败: {e}")
            return self._get_fallback_constraints()
    
    def _get_fallback_constraints(self) -> Dict[str, Dict[str, Constraint]]:
        """获取备用约束（当JSON文件不可用时）"""
        self.logger.info("使用内置备用约束")
        return {
            "data_parallel": {
                "DP权重一致性检查": Constraint(
                    name="DP权重一致性检查",
                    description="检查数据并行训练中相同权重在不同DP rank间的一致性",
                    type=ConstraintType.CONSISTENCY,
                    logic="WITH dp_weights AS (SELECT name, dp, AVG(cksum) as avg_cksum, STDDEV(cksum) as std_cksum FROM coredump WHERE type = 'weight' GROUP BY name, dp) SELECT COUNT(*) as inconsistent_weights FROM dp_weights WHERE std_cksum > 0.001",
                    tables=["coredump"],
                    params={"threshold": 0.001}
                )
            }
        }
    
    def get_constraints_by_category(self, category: str) -> List[Constraint]:
        """根据类别获取约束"""
        category_constraints = self._constraints.get(category, {})
        return list(category_constraints.values())
    
    def get_all_categories(self) -> List[str]:
        """获取所有约束类别"""
        return list(self._constraints.keys())
    
    def get_constraints_for_config(self, config: TrainingConfig) -> List[Constraint]:
        """根据训练配置自动选择适用的约束"""
        applicable_constraints = []
        
        # 根据配置选择相关约束
        if config.dp > 1:
            applicable_constraints.extend(self.get_constraints_by_category("data_parallel"))
        
        if config.tp > 1:
            applicable_constraints.extend(self.get_constraints_by_category("tensor_parallel"))
        
        if config.pp > 1:
            applicable_constraints.extend(self.get_constraints_by_category("pipeline_parallel"))
        
        if config.zero_stage > 0:
            applicable_constraints.extend(self.get_constraints_by_category("zero_optimization"))
        
        # 总是包含模型完整性和训练进度检查
        applicable_constraints.extend(self.get_constraints_by_category("model_integrity"))
        applicable_constraints.extend(self.get_constraints_by_category("training_progress"))
        
        return applicable_constraints
    
    def get_constraint_by_name(self, name: str) -> Constraint:
        """根据名称获取特定约束"""
        for category_constraints in self._constraints.values():
            if name in category_constraints:
                return category_constraints[name]
        raise ValueError(f"未找到名为 '{name}' 的约束")
    
    def add_custom_constraint(self, category: str, constraint: Constraint) -> None:
        """添加自定义约束到指定类别"""
        if category not in self._constraints:
            self._constraints[category] = {}
        self._constraints[category][constraint.name] = constraint
    
    def list_all_constraints(self) -> Dict[str, List[str]]:
        """列出所有约束的名称，按类别组织"""
        result = {}
        for category, constraints in self._constraints.items():
            result[category] = list(constraints.keys())
        return result
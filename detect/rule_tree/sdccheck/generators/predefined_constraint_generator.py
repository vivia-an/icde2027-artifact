"""预定义约束生成器，使用预存储的约束规则而非LLM生成"""

from typing import List, Dict, Any, Optional
import logging

from ..models import SchemaInfo, TrainingConfig, Constraint
from ..predefined_constraints import PredefinedConstraints


class PredefinedConstraintGenerator:
    """预定义约束生成器，从预存储的约束库中选择适用的约束"""
    
    def __init__(self, custom_constraints: Optional[Dict[str, List[Constraint]]] = None,
                 config_path: Optional[str] = None):
        """
        初始化预定义约束生成器
        
        参数:
            custom_constraints: 自定义约束字典，可选
            config_path: 约束配置文件路径，如果提供则使用该文件，否则使用默认配置
        """
        self.predefined_constraints = PredefinedConstraints(config_path=config_path)
        self.logger = logging.getLogger("SDCCheck.PredefinedConstraintGenerator")
        
        # 添加自定义约束
        if custom_constraints:
            for category, constraints in custom_constraints.items():
                for constraint in constraints:
                    self.predefined_constraints.add_custom_constraint(category, constraint)
    
    def generate(self, schema_info: SchemaInfo, config: TrainingConfig, 
                 categories: Optional[List[str]] = None,
                 constraint_names: Optional[List[str]] = None) -> List[Constraint]:
        """
        根据数据库模式和训练配置生成约束列表
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置，包含DP/TP等并行信息
            categories: 指定的约束类别列表，如果为None则自动选择
            constraint_names: 指定的约束名称列表，如果提供则只返回这些约束
            
        返回:
            List[Constraint]: 选择的约束列表
        """
        self.logger.info(f"开始生成预定义约束，DP={config.dp}, TP={config.tp}, PP={config.pp}")
        
        constraints = []
        
        # 如果指定了具体的约束名称，直接返回这些约束
        if constraint_names:
            for name in constraint_names:
                try:
                    constraint = self.predefined_constraints.get_constraint_by_name(name)
                    # 验证约束是否适用于当前数据库模式
                    if self._is_constraint_applicable(constraint, schema_info):
                        constraints.append(self._adapt_constraint_to_schema(constraint, schema_info, config))
                    else:
                        self.logger.warning(f"约束 '{name}' 不适用于当前数据库模式，已跳过")
                except ValueError as e:
                    self.logger.error(f"未找到约束 '{name}': {e}")
            return constraints
        
        # 如果指定了类别，使用指定类别
        if categories:
            for category in categories:
                category_constraints = self.predefined_constraints.get_constraints_by_category(category)
                for constraint in category_constraints:
                    if self._is_constraint_applicable(constraint, schema_info):
                        constraints.append(self._adapt_constraint_to_schema(constraint, schema_info, config))
        else:
            # 自动根据训练配置选择约束
            constraints = self.predefined_constraints.get_constraints_for_config(config)
            # 过滤并适配约束
            applicable_constraints = []
            for constraint in constraints:
                if self._is_constraint_applicable(constraint, schema_info):
                    applicable_constraints.append(self._adapt_constraint_to_schema(constraint, schema_info, config))
                else:
                    self.logger.debug(f"约束 '{constraint.name}' 不适用于当前数据库模式，已跳过")
            constraints = applicable_constraints
        
        self.logger.info(f"成功生成{len(constraints)}个预定义约束")
        
        # 记录生成的约束信息
        for constraint in constraints:
            self.logger.debug(f"选择约束: {constraint.name} ({constraint.type})")
        
        return constraints
    
    def _is_constraint_applicable(self, constraint: Constraint, schema_info: SchemaInfo) -> bool:
        """
        检查约束是否适用于当前数据库模式
        
        参数:
            constraint: 约束对象
            schema_info: 数据库模式信息
            
        返回:
            bool: 是否适用
        """
        try:
            # 1. 检查约束涉及的表是否存在
            for table in constraint.tables:
                if table not in schema_info.tables:
                    self.logger.debug(f"约束 '{constraint.name}' 需要的表 '{table}' 不存在")
                    return False
            
            # 2. 检查约束逻辑中涉及的列是否存在
            if not self._check_required_columns(constraint, schema_info):
                return False
            
            # 3. 检查约束逻辑中涉及的JSON字段是否存在
            if not self._check_required_json_fields(constraint, schema_info):
                return False
            
            # 4. 检查约束的适用条件是否满足
            if not self._check_applicable_conditions(constraint, schema_info):
                return False
            
            self.logger.debug(f"约束 '{constraint.name}' 适用于当前数据库模式")
            return True
            
        except Exception as e:
            self.logger.error(f"检查约束 '{constraint.name}' 适用性时发生错误: {e}")
            return False
    
    def _adapt_constraint_to_schema(self, constraint: Constraint, 
                                   schema_info: SchemaInfo, 
                                   config: TrainingConfig) -> Constraint:
        """
        根据实际的数据库模式和配置调整约束
        
        参数:
            constraint: 原始约束
            schema_info: 数据库模式信息
            config: 训练配置
            
        返回:
            Constraint: 调整后的约束
        """
        # 创建约束的副本
        adapted_constraint = Constraint(
            name=constraint.name,
            description=constraint.description,
            type=constraint.type,
            logic=constraint.logic,
            tables=constraint.tables.copy(),
            params=constraint.params.copy(),
            applicable_conditions=constraint.applicable_conditions.copy()
        )
        
        # 根据实际配置调整参数
        if "expected_dp_count" in adapted_constraint.params or "min_dp_count" in adapted_constraint.params:
            adapted_constraint.params["expected_dp_count"] = config.dp
        
        if "expected_tp_count" in adapted_constraint.params or "min_tp_count" in adapted_constraint.params:
            adapted_constraint.params["expected_tp_count"] = config.tp
        
        if "expected_pp_count" in adapted_constraint.params:
            adapted_constraint.params["expected_pp_count"] = config.pp
        
        if "zero_stage" in adapted_constraint.params:
            adapted_constraint.params["zero_stage"] = config.zero_stage
        
        # 如果约束涉及的表名是通用的，替换为实际的表名
        if "coredump" in adapted_constraint.tables and "coredump" not in schema_info.tables:
            # 尝试找到实际的表名
            actual_tables = list(schema_info.tables.keys())
            if actual_tables:
                adapted_constraint.tables = [actual_tables[0]]  # 使用第一个找到的表
                self.logger.debug(f"将约束 '{constraint.name}' 的表名从 'coredump' 调整为 '{actual_tables[0]}'")
        
        return adapted_constraint
    
    def _check_required_columns(self, constraint: Constraint, schema_info: SchemaInfo) -> bool:
        """
        检查约束逻辑中涉及的列是否存在于数据库模式中
        
        参数:
            constraint: 约束对象
            schema_info: 数据库模式信息
            
        返回:
            bool: 所有必需列是否存在
        """
        import re
        
        # 从约束逻辑中提取可能的列名
        logic = constraint.logic.lower()
        
        # 常见的列名模式
        common_columns = [
            'name', 'dp', 'tp', 'pp', 'ep', 'sp', 'step', 'stage', 'data',
            'cksum', 'grad_cksum', 'shape', 'type', 'requires_grad'
        ]
        
        # 检查每个表中是否包含必需的列
        for table_name in constraint.tables:
            if table_name not in schema_info.tables:
                continue
                
            table_info = schema_info.tables[table_name]
            table_columns = [col['name'].lower() for col in table_info.get('columns', [])]
            
            # 检查逻辑中明确引用的列
            for column in common_columns:
                if column in logic and column not in table_columns:
                    # 检查是否是JSON字段中的键
                    if not self._is_json_field_available(column, table_info):
                        self.logger.debug(f"约束 '{constraint.name}' 需要的列 '{column}' 在表 '{table_name}' 中不存在")
                        return False
        
        return True
    
    def _check_required_json_fields(self, constraint: Constraint, schema_info: SchemaInfo) -> bool:
        """
        检查约束逻辑中涉及的JSON字段是否存在
        
        参数:
            constraint: 约束对象
            schema_info: 数据库模式信息
            
        返回:
            bool: 所有必需JSON字段是否存在
        """
        import re
        
        logic = constraint.logic
        
        # 查找JSON_EXTRACT相关的字段引用
        json_extract_pattern = r"JSON_EXTRACT(?:_STRING)?\([^,]+,\s*['\"]\$\.([^'\"]+)['\"]\)"
        json_fields = re.findall(json_extract_pattern, logic, re.IGNORECASE)
        
        if not json_fields:
            return True  # 没有JSON字段要求
        
        # 检查每个表中是否包含必需的JSON字段
        for table_name in constraint.tables:
            if table_name not in schema_info.tables:
                continue
                
            table_info = schema_info.tables[table_name]
            
            for json_field in json_fields:
                if not self._is_json_field_available(json_field, table_info):
                    self.logger.debug(f"约束 '{constraint.name}' 需要的JSON字段 '{json_field}' 在表 '{table_name}' 中不存在")
                    return False
        
        return True
    
    def _is_json_field_available(self, field_name: str, table_info: Dict[str, Any]) -> bool:
        """
        检查指定的字段是否在表的JSON列中可用
        
        参数:
            field_name: 字段名
            table_info: 表信息
            
        返回:
            bool: 字段是否可用
        """
        json_keys = table_info.get('json_keys', {})
        
        for column_name, keys in json_keys.items():
            if field_name in keys:
                return True
        
        return False
    
    def _check_applicable_conditions(self, constraint: Constraint, schema_info: SchemaInfo) -> bool:
        """
        检查约束的适用条件是否满足
        
        参数:
            constraint: 约束对象
            schema_info: 数据库模式信息
            
        返回:
            bool: 适用条件是否满足
        """
        # 检查是否有_skip标记（动态注入的跳过条件）
        if constraint.applicable_conditions.get('_skip') == "= true":
            self.logger.debug(f"约束 '{constraint.name}' 被动态标记为跳过")
            return False
        
        # 检查所有适用条件
        for condition_key, condition_value in constraint.applicable_conditions.items():
            if condition_key == '_skip':
                continue  # 已经处理过了
                
            # 检查具体的条件类型
            if not self._evaluate_single_condition(condition_key, condition_value, schema_info):
                self.logger.debug(f"约束 '{constraint.name}' 的条件 '{condition_key}: {condition_value}' 不满足")
                return False
        
        # 检查约束名称中的特定模式
        constraint_name = constraint.name.lower()
        
        # 如果约束涉及特定的权重类型，检查是否存在相关数据
        weight_patterns = {
            'embedding': ['embedding', 'word_embeddings'],
            'attention': ['attention', 'qkv', 'proj'],
            'mlp': ['mlp', 'fc1', 'fc2', 'linear'],
            'layernorm': ['layernorm', 'layer_norm', 'norm']
        }
        
        for pattern_type, patterns in weight_patterns.items():
            if any(pattern in constraint_name for pattern in patterns):
                # 检查是否存在相关的权重数据
                if not self._has_relevant_weight_data(patterns, schema_info):
                    self.logger.debug(f"约束 '{constraint.name}' 需要的权重类型 '{pattern_type}' 在数据库中不存在")
                    return False
        
        return True
    
    def _evaluate_single_condition(self, condition_key: str, condition_value: str, schema_info: SchemaInfo) -> bool:
        """
        评估单个条件是否满足
        
        参数:
            condition_key: 条件键（如'dp', 'tp', 'stage'等）
            condition_value: 条件值（如'> 1', '= model-after-optimizer-step'）
            schema_info: 数据库模式信息
            
        返回:
            bool: 条件是否满足
        """
        # 这里需要根据实际的数据库内容来评估条件
        # 目前为了保持兼容性，我们先返回True
        # 在实际使用中，这些条件应该在运行时通过查询数据库来验证
        
        if condition_key in ['dp', 'tp', 'pp', 'ep']:
            # 并行度条件已经在script_config_injector中处理了
            # 如果代码执行到这里，说明条件已经满足或者没有配置脚本注入
            return True
            
        elif condition_key == 'stage':
            # stage条件需要在运行时验证，这里先返回True
            return True
            
        elif condition_key == 'parallel_groups':
            # parallel_groups条件需要在运行时验证，这里先返回True  
            return True
            
        else:
            # 其他未知条件，保守地返回True
            self.logger.warning(f"未知的条件键: {condition_key}")
            return True
    
    def _has_relevant_weight_data(self, patterns: List[str], schema_info: SchemaInfo) -> bool:
        """
        检查数据库中是否存在相关的权重数据
        
        参数:
            patterns: 权重名称模式列表
            schema_info: 数据库模式信息
            
        返回:
            bool: 是否存在相关权重数据
        """
        # 这里可以实现更复杂的数据存在性检查
        # 目前简化为检查是否有name字段或相关JSON键
        
        for table_name, table_info in schema_info.tables.items():
            # 检查是否有name列
            columns = [col['name'].lower() for col in table_info.get('columns', [])]
            if 'name' in columns:
                return True
            
            # 检查JSON字段中是否有name键
            json_keys = table_info.get('json_keys', {})
            for column_name, keys in json_keys.items():
                if 'name' in keys:
                    return True
        
        return False
    
    def list_available_constraints(self) -> Dict[str, List[str]]:
        """
        列出所有可用的约束
        
        返回:
            Dict[str, List[str]]: 按类别组织的约束名称列表
        """
        return self.predefined_constraints.list_all_constraints()
    
    def get_constraints_by_category(self, category: str) -> List[Constraint]:
        """
        获取指定类别的所有约束
        
        参数:
            category: 约束类别
            
        返回:
            List[Constraint]: 约束列表
        """
        return self.predefined_constraints.get_constraints_by_category(category)
    
    def add_custom_constraint(self, category: str, constraint: Constraint) -> None:
        """
        添加自定义约束
        
        参数:
            category: 约束类别
            constraint: 约束对象
        """
        self.predefined_constraints.add_custom_constraint(category, constraint)
        self.logger.info(f"添加自定义约束 '{constraint.name}' 到类别 '{category}'")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        获取生成器统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        constraints_by_category = self.list_available_constraints()
        total_constraints = sum(len(constraints) for constraints in constraints_by_category.values())
        
        # 统计自定义约束数量（假设自定义约束在特定类别中）
        custom_categories = ["custom", "user_defined", "performance"]  # 可能的自定义类别
        custom_count = sum(
            len(constraints_by_category.get(category, []))
            for category in custom_categories
            if category in constraints_by_category
        )
        
        return {
            "total_constraints": total_constraints,
            "categories": list(constraints_by_category.keys()),
            "category_counts": {cat: len(constraints) for cat, constraints in constraints_by_category.items()},
            "custom_constraints_count": custom_count,
            "generator_type": "predefined"
        }
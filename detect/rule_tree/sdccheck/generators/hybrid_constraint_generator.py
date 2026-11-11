"""混合约束生成器，支持LLM生成和预定义约束两种方式"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

from ..models import SchemaInfo, TrainingConfig, Constraint
from ..config import LLMConfig
from .llm_constraint_generator import LLMConstraintGenerator
from .predefined_constraint_generator import PredefinedConstraintGenerator


class ConstraintGenerationMode(str, Enum):
    """约束生成模式枚举"""
    LLM = "llm"  # 使用LLM生成
    PREDEFINED = "predefined"  # 使用预定义约束
    HYBRID = "hybrid"  # 混合模式：优先使用预定义，不足时用LLM补充


class HybridConstraintGenerator:
    """混合约束生成器，支持多种约束生成策略"""
    
    def __init__(self, 
                 llm_config: Optional[LLMConfig] = None, 
                 provider_name: str = "mock",
                 custom_constraints: Optional[Dict[str, List[Constraint]]] = None,
                 default_mode: ConstraintGenerationMode = ConstraintGenerationMode.PREDEFINED):
        """
        初始化混合约束生成器
        
        参数:
            llm_config: LLM配置，用于LLM生成模式
            provider_name: LLM提供商名称
            custom_constraints: 自定义约束字典
            default_mode: 默认生成模式
        """
        self.default_mode = default_mode
        self.logger = logging.getLogger("SDCCheck.HybridConstraintGenerator")
        
        # 初始化预定义约束生成器
        self.predefined_generator = PredefinedConstraintGenerator(custom_constraints)
        
        # 初始化LLM约束生成器
        self.llm_generator = None
        if llm_config:
            self.llm_generator = LLMConstraintGenerator(llm_config, provider_name)
        
        self.logger.info(f"初始化混合约束生成器，默认模式: {default_mode}")
    
    def generate(self, 
                 schema_info: SchemaInfo, 
                 config: TrainingConfig,
                 mode: Optional[ConstraintGenerationMode] = None,
                 predefined_options: Optional[Dict[str, Any]] = None,
                 llm_fallback: bool = True) -> List[Constraint]:
        """
        根据指定模式生成约束列表
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置
            mode: 生成模式，如果为None则使用默认模式
            predefined_options: 预定义约束选项，包含categories和constraint_names
            llm_fallback: 在预定义约束不足时是否回退到LLM生成
            
        返回:
            List[Constraint]: 生成的约束列表
        """
        if mode is None:
            mode = self.default_mode
        
        self.logger.info(f"开始生成约束，模式: {mode}")
        
        if mode == ConstraintGenerationMode.PREDEFINED:
            return self._generate_predefined(schema_info, config, predefined_options)
        
        elif mode == ConstraintGenerationMode.LLM:
            return self._generate_llm(schema_info, config)
        
        elif mode == ConstraintGenerationMode.HYBRID:
            return self._generate_hybrid(schema_info, config, predefined_options, llm_fallback)
        
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def _generate_predefined(self, 
                           schema_info: SchemaInfo, 
                           config: TrainingConfig,
                           options: Optional[Dict[str, Any]] = None) -> List[Constraint]:
        """
        使用预定义约束生成
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置
            options: 预定义约束选项
            
        返回:
            List[Constraint]: 约束列表
        """
        self.logger.info("使用预定义约束生成")
        
        if options is None:
            options = {}
        
        categories = options.get("categories")
        constraint_names = options.get("constraint_names")
        
        constraints = self.predefined_generator.generate(
            schema_info, config, categories, constraint_names
        )
        
        self.logger.info(f"预定义约束生成完成，共{len(constraints)}个约束")
        return constraints
    
    def _generate_llm(self, schema_info: SchemaInfo, config: TrainingConfig) -> List[Constraint]:
        """
        使用LLM生成约束
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置
            
        返回:
            List[Constraint]: 约束列表
        """
        if self.llm_generator is None:
            raise ValueError("LLM生成器未初始化，请提供LLM配置")
        
        self.logger.info("使用LLM生成约束")
        constraints = self.llm_generator.generate(schema_info, config)
        self.logger.info(f"LLM约束生成完成，共{len(constraints)}个约束")
        return constraints
    
    def _generate_hybrid(self, 
                        schema_info: SchemaInfo, 
                        config: TrainingConfig,
                        predefined_options: Optional[Dict[str, Any]] = None,
                        llm_fallback: bool = True) -> List[Constraint]:
        """
        使用混合模式生成约束
        
        参数:
            schema_info: 数据库模式信息
            config: 训练配置
            predefined_options: 预定义约束选项
            llm_fallback: 是否在预定义约束不足时使用LLM补充
            
        返回:
            List[Constraint]: 约束列表
        """
        self.logger.info("使用混合模式生成约束")
        
        # 首先尝试预定义约束
        constraints = self._generate_predefined(schema_info, config, predefined_options)
        
        # 如果预定义约束数量不足且启用了LLM回退
        min_constraints = 3  # 最少约束数量
        if len(constraints) < min_constraints and llm_fallback and self.llm_generator:
            self.logger.info(f"预定义约束数量({len(constraints)})不足，使用LLM补充")
            
            try:
                llm_constraints = self._generate_llm(schema_info, config)
                
                # 去重：避免添加重复的约束
                existing_names = {c.name for c in constraints}
                for llm_constraint in llm_constraints:
                    if llm_constraint.name not in existing_names:
                        constraints.append(llm_constraint)
                        existing_names.add(llm_constraint.name)
                
                self.logger.info(f"LLM补充完成，总约束数量: {len(constraints)}")
                
            except Exception as e:
                self.logger.warning(f"LLM补充失败: {e}，继续使用预定义约束")
        
        return constraints
    
    def list_available_predefined_constraints(self) -> Dict[str, List[str]]:
        """
        列出所有可用的预定义约束
        
        返回:
            Dict[str, List[str]]: 按类别组织的约束名称列表
        """
        return self.predefined_generator.list_available_constraints()
    
    def get_predefined_constraints_by_category(self, category: str) -> List[Constraint]:
        """
        获取指定类别的预定义约束
        
        参数:
            category: 约束类别
            
        返回:
            List[Constraint]: 约束列表
        """
        return self.predefined_generator.get_constraints_by_category(category)
    
    def add_custom_constraint(self, category: str, constraint: Constraint) -> None:
        """
        添加自定义约束到预定义约束库
        
        参数:
            category: 约束类别
            constraint: 约束对象
        """
        self.predefined_generator.add_custom_constraint(category, constraint)
        self.logger.info(f"添加自定义约束 '{constraint.name}' 到类别 '{category}'")
    
    def update_llm_config(self, llm_config: LLMConfig, provider_name: str = None) -> None:
        """
        更新LLM配置
        
        参数:
            llm_config: 新的LLM配置
            provider_name: 新的提供商名称（可选）
        """
        if self.llm_generator:
            self.llm_generator.update_llm_config(llm_config, provider_name)
        else:
            # 如果LLM生成器不存在，创建一个新的
            self.llm_generator = LLMConstraintGenerator(
                llm_config, provider_name or "mock"
            )
        
        self.logger.info("LLM配置已更新")
    
    def set_default_mode(self, mode: ConstraintGenerationMode) -> None:
        """
        设置默认生成模式
        
        参数:
            mode: 新的默认模式
        """
        self.default_mode = mode
        self.logger.info(f"默认生成模式已设置为: {mode}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        获取约束生成统计信息
        
        返回:
            Dict[str, Any]: 生成统计信息
        """
        predefined_stats = self.predefined_generator.list_available_constraints()
        total_predefined = sum(len(constraints) for constraints in predefined_stats.values())
        
        stats = {
            "current_mode": self.default_mode.value,
            "supported_modes": [mode.value for mode in ConstraintGenerationMode],
            "predefined_constraints_count": total_predefined,
            "predefined_categories": list(predefined_stats.keys()),
            "has_llm_fallback": self.llm_generator is not None,
            "custom_constraints_count": 0  # 可以从predefined_generator获取
        }
        
        if self.llm_generator and hasattr(self.llm_generator, 'get_generation_stats'):
            stats["llm_generator_available"] = True
        else:
            stats["llm_generator_available"] = False
            
        return stats
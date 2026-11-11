#!/usr/bin/env python3
"""
脚本配置注入器
从Megatron训练脚本中解析参数，并动态生成适用的约束配置文件
"""

import json
import re
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .models import TrainingConfig


@dataclass
class ScriptConfig:
    """从脚本解析的训练配置"""
    # 并行配置
    tp: int = 1  # tensor parallel
    pp: int = 1  # pipeline parallel  
    dp: int = 1  # data parallel (计算得出)
    ep: int = 1  # expert parallel
    sp: int = 1  # sequence parallel
    cp: int = 1  # context parallel
    
    # 优化器配置
    use_distributed_optimizer: bool = False
    zero_stage: int = 0
    
    # 模型配置
    num_layers: int = 12
    hidden_size: int = 128
    num_attention_heads: int = 4
    seq_length: int = 64
    micro_batch_size: int = 2
    global_batch_size: int = 16
    
    # 训练配置
    train_iters: int = 100
    sequence_parallel: bool = False
    untie_embeddings: bool = False
    
    # 运行时配置
    world_size: int = 1
    nproc_per_node: int = 1
    nnodes: int = 1


class ScriptConfigInjector:
    """脚本配置注入器，解析训练脚本并生成动态约束配置"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"SDCCheck.{self.__class__.__name__}")
        
    def parse_script(self, script_path: str) -> ScriptConfig:
        """
        解析Megatron训练脚本，提取所有相关配置
        
        参数:
            script_path: 训练脚本路径
            
        返回:
            ScriptConfig: 解析得到的完整配置
        """
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"训练脚本文件不存在: {script_path}")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config = ScriptConfig()
        
        # 解析并行配置
        self._parse_parallel_config(content, config)
        
        # 解析模型配置
        self._parse_model_config(content, config)
        
        # 解析训练配置
        self._parse_training_config(content, config)
        
        # 解析运行时配置
        self._parse_runtime_config(content, config)
        
        # 计算派生配置
        self._calculate_derived_config(config)
        
        return config
    
    def _parse_parallel_config(self, content: str, config: ScriptConfig) -> None:
        """解析并行配置参数"""
        # 张量并行
        tp_patterns = [
            r'--tensor-model-parallel-size\s+(\d+)',
            r'TP_SIZE=(\d+)',
            r'TP_SIZE=\$\{TP_SIZE:-(\d+)\}'
        ]
        config.tp = self._extract_int_value(content, tp_patterns, config.tp)
        
        # 流水线并行
        pp_patterns = [
            r'--pipeline-model-parallel-size\s+(\d+)',
            r'PP_SIZE=(\d+)',
            r'PP_SIZE=\$\{PP_SIZE:-(\d+)\}'
        ]
        config.pp = self._extract_int_value(content, pp_patterns, config.pp)
        
        # 专家并行
        ep_patterns = [
            r'--expert-model-parallel-size\s+(\d+)',
            r'EP_SIZE=(\d+)',
            r'EP_SIZE=\$\{EP_SIZE:-(\d+)\}'
        ]
        config.ep = self._extract_int_value(content, ep_patterns, config.ep)
        
        # 上下文并行
        cp_patterns = [
            r'--context-parallel-size\s+(\d+)',
            r'CP_SIZE=(\d+)',
            r'CP_SIZE=\$\{CP_SIZE:-(\d+)\}'
        ]
        config.cp = self._extract_int_value(content, cp_patterns, config.cp)
        
        # 序列并行
        config.sequence_parallel = '--sequence-parallel' in content
        if config.sequence_parallel:
            config.sp = max(config.tp, 1)  # 序列并行通常与TP绑定
    
    def _parse_model_config(self, content: str, config: ScriptConfig) -> None:
        """解析模型配置参数"""
        # 模型层数
        config.num_layers = self._extract_int_value(
            content, [r'--num-layers\s+(\d+)'], config.num_layers
        )
        
        # 隐藏层大小
        config.hidden_size = self._extract_int_value(
            content, [r'--hidden-size\s+(\d+)'], config.hidden_size
        )
        
        # 注意力头数
        config.num_attention_heads = self._extract_int_value(
            content, [r'--num-attention-heads\s+(\d+)'], config.num_attention_heads
        )
        
        # 序列长度
        config.seq_length = self._extract_int_value(
            content, [r'--seq-length\s+(\d+)'], config.seq_length
        )
        
        # 解绑嵌入层权重
        config.untie_embeddings = '--untie-embeddings-and-output-weights' in content
    
    def _parse_training_config(self, content: str, config: ScriptConfig) -> None:
        """解析训练配置参数"""
        # 批次大小
        config.micro_batch_size = self._extract_int_value(
            content, [r'--micro-batch-size\s+(\d+)'], config.micro_batch_size
        )
        
        config.global_batch_size = self._extract_int_value(
            content, [r'--global-batch-size\s+(\d+)'], config.global_batch_size
        )
        
        # 训练迭代次数
        config.train_iters = self._extract_int_value(
            content, [r'--train-iters\s+(\d+)'], config.train_iters
        )
        
        # 分布式优化器
        config.use_distributed_optimizer = '--use-distributed-optimizer' in content
        if config.use_distributed_optimizer:
            config.zero_stage = 1  # 使用分布式优化器通常对应ZeRO stage 1
    
    def _parse_runtime_config(self, content: str, config: ScriptConfig) -> None:
        """解析运行时配置参数"""
        # nproc_per_node
        config.nproc_per_node = self._extract_int_value(
            content, [r'--nproc_per_node=(\d+)', r'--nproc-per-node\s+(\d+)'], config.nproc_per_node
        )
        
        # nnodes
        config.nnodes = self._extract_int_value(
            content, [r'--nnodes=(\d+)', r'--nnodes\s+(\d+)'], config.nnodes
        )
    
    def _calculate_derived_config(self, config: ScriptConfig) -> None:
        """计算派生配置"""
        # 计算world_size
        config.world_size = config.nproc_per_node * config.nnodes
        
        # 计算data parallel大小
        config.dp = config.world_size // (config.tp * config.pp * config.ep)
        config.dp = max(1, config.dp)  # 确保至少为1
    
    def _extract_int_value(self, content: str, patterns: List[str], default: int) -> int:
        """从内容中提取整数值"""
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return int(match.group(1))
        return default
    
    def generate_dynamic_constraints_config(self, script_config: ScriptConfig, 
                                           base_config_path: str) -> Dict[str, Any]:
        """
        根据脚本配置生成动态约束配置
        
        参数:
            script_config: 从脚本解析的配置
            base_config_path: 基础约束配置文件路径
            
        返回:
            Dict: 动态生成的约束配置
        """
        # 加载基础约束配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        
        # 创建动态配置副本
        dynamic_config = base_config.copy()
        
        # 注入脚本参数到约束条件中
        self._inject_script_params_to_constraints(dynamic_config, script_config)
        
        # 添加元数据
        dynamic_config['metadata']['generated_from_script'] = True
        dynamic_config['metadata']['script_config'] = {
            'tp': script_config.tp,
            'pp': script_config.pp,
            'dp': script_config.dp,
            'ep': script_config.ep,
            'sp': script_config.sp,
            'cp': script_config.cp,
            'use_distributed_optimizer': script_config.use_distributed_optimizer,
            'zero_stage': script_config.zero_stage,
            'sequence_parallel': script_config.sequence_parallel,
            'untie_embeddings': script_config.untie_embeddings
        }
        
        return dynamic_config
    
    def _inject_script_params_to_constraints(self, config: Dict[str, Any], 
                                           script_config: ScriptConfig) -> None:
        """将脚本参数注入到约束条件中"""
        constraints = config.get('constraints', {})
        
        for category, constraint_dict in constraints.items():
            for constraint_name, constraint_info in constraint_dict.items():
                applicable_conditions = constraint_info.get('applicable_conditions', {})
                
                # 注入并行度条件
                self._inject_parallel_conditions(applicable_conditions, script_config)
                
                # 注入模型特定条件
                self._inject_model_conditions(applicable_conditions, script_config, constraint_name)
                
                # 注入训练特定条件
                self._inject_training_conditions(applicable_conditions, script_config, constraint_name)
    
    def _inject_parallel_conditions(self, conditions: Dict[str, str], 
                                   script_config: ScriptConfig) -> None:
        """注入并行度相关条件"""
        # 评估原始条件，如果不满足则标记跳过
        should_skip = False
        
        # 检查DP条件
        if 'dp' in conditions:
            original_condition = conditions['dp']
            if not self._evaluate_condition('dp', original_condition, script_config.dp):
                should_skip = True
                
        # 检查TP条件  
        if 'tp' in conditions:
            original_condition = conditions['tp']
            if not self._evaluate_condition('tp', original_condition, script_config.tp):
                should_skip = True
                
        # 检查PP条件
        if 'pp' in conditions:
            original_condition = conditions['pp']
            if not self._evaluate_condition('pp', original_condition, script_config.pp):
                should_skip = True
                
        # 如果任何并行度条件不满足，标记跳过整个约束
        if should_skip:
            conditions['_skip'] = "= true"
    
    def _evaluate_condition(self, param_name: str, condition: str, actual_value: int) -> bool:
        """
        评估条件是否满足
        
        参数:
            param_name: 参数名（如'dp', 'tp', 'pp'）
            condition: 条件字符串（如'> 1', '>= 2', '= 1'）
            actual_value: 实际值
            
        返回:
            bool: 条件是否满足
        """
        try:
            # 解析条件
            condition = condition.strip()
            
            if condition.startswith('> '):
                threshold = int(condition[2:].strip())
                return actual_value > threshold
            elif condition.startswith('>= '):
                threshold = int(condition[3:].strip())
                return actual_value >= threshold
            elif condition.startswith('= '):
                expected = int(condition[2:].strip())
                return actual_value == expected
            elif condition.startswith('!= '):
                not_expected = int(condition[3:].strip())
                return actual_value != not_expected
            elif condition.startswith('< '):
                threshold = int(condition[2:].strip())
                return actual_value < threshold
            elif condition.startswith('<= '):
                threshold = int(condition[3:].strip())
                return actual_value <= threshold
            else:
                # 如果条件格式不识别，假设条件满足（保守处理）
                self.logger.warning(f"未识别的条件格式: {condition}")
                return True
                
        except (ValueError, IndexError) as e:
            self.logger.error(f"解析条件 '{condition}' 时出错: {e}")
            return True  # 解析失败时保守处理，假设条件满足
    
    def _inject_model_conditions(self, conditions: Dict[str, str], 
                                script_config: ScriptConfig, constraint_name: str) -> None:
        """注入模型特定条件"""
        # 权重共享相关约束
        if 'weight_tying' in constraint_name.lower() or '权重共享' in constraint_name:
            if not script_config.untie_embeddings:
                # 如果没有解绑嵌入层，启用权重共享检查
                conditions['untie_embeddings'] = "= false"
            else:
                # 如果解绑了嵌入层，禁用权重共享检查
                conditions['untie_embeddings'] = "= true"
                conditions['_skip'] = "= true"  # 标记跳过
        
        # 序列并行相关约束
        if 'sequence_parallel' in constraint_name.lower() or '序列并行' in constraint_name:
            if not script_config.sequence_parallel:
                conditions['_skip'] = "= true"
    
    def _inject_training_conditions(self, conditions: Dict[str, str], 
                                   script_config: ScriptConfig, constraint_name: str) -> None:
        """注入训练特定条件"""
        # ZeRO优化器相关约束
        if 'zero' in constraint_name.lower() or 'distributed_optimizer' in constraint_name.lower():
            if not script_config.use_distributed_optimizer:
                conditions['_skip'] = "= true"
            else:
                conditions['zero_stage'] = f">= {script_config.zero_stage}"
    
    def save_dynamic_config(self, dynamic_config: Dict[str, Any], output_path: str) -> None:
        """保存动态生成的约束配置，自动过滤掉标记为跳过的约束"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 过滤掉标记为跳过的约束
        filtered_config = self._filter_skipped_constraints(dynamic_config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"动态约束配置已保存到: {output_path}")
    
    def _filter_skipped_constraints(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤掉标记为跳过的约束
        
        参数:
            config: 原始配置字典
            
        返回:
            Dict: 过滤后的配置字典
        """
        filtered_config = config.copy()
        constraints = filtered_config.get('constraints', {})
        
        original_count = 0
        filtered_count = 0
        
        for category, constraint_dict in constraints.items():
            filtered_constraints = {}
            
            for constraint_name, constraint_info in constraint_dict.items():
                original_count += 1
                
                # 检查是否有_skip标记
                applicable_conditions = constraint_info.get('applicable_conditions', {})
                
                if applicable_conditions.get('_skip') == "= true":
                    self.logger.debug(f"过滤掉标记为跳过的约束: {constraint_name}")
                    continue
                else:
                    filtered_constraints[constraint_name] = constraint_info
                    filtered_count += 1
            
            constraints[category] = filtered_constraints
        
        self.logger.info(f"约束过滤完成: 原始数量={original_count}, 过滤后数量={filtered_count}, "
                        f"过滤掉数量={original_count - filtered_count}")
        
        return filtered_config
    
    def inject_script_config_to_constraints(self, script_path: str, 
                                          base_config_path: str,
                                          output_path: str) -> ScriptConfig:
        """
        完整的脚本配置注入流程
        
        参数:
            script_path: 训练脚本路径
            base_config_path: 基础约束配置文件路径  
            output_path: 输出的动态约束配置文件路径
            
        返回:
            ScriptConfig: 解析得到的脚本配置
        """
        # 解析脚本配置
        script_config = self.parse_script(script_path)
        
        self.logger.info(f"从脚本解析的配置: TP={script_config.tp}, PP={script_config.pp}, "
                        f"DP={script_config.dp}, EP={script_config.ep}, SP={script_config.sp}")
        
        # 生成动态约束配置
        dynamic_config = self.generate_dynamic_constraints_config(script_config, base_config_path)
        
        # 保存动态配置
        self.save_dynamic_config(dynamic_config, output_path)
        
        return script_config


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="脚本配置注入器")
    parser.add_argument('script_path', help='Megatron训练脚本路径')
    parser.add_argument('--base-config', default='config/predefined_constraints.json',
                       help='基础约束配置文件路径')
    parser.add_argument('--output', default='config/dynamic_constraints.json',
                       help='输出的动态约束配置文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建注入器并执行
    injector = ScriptConfigInjector()
    script_config = injector.inject_script_config_to_constraints(
        args.script_path,
        args.base_config,
        args.output
    )
    
    print(f"脚本配置注入完成!")
    print(f"解析的配置: TP={script_config.tp}, PP={script_config.pp}, DP={script_config.dp}")
    print(f"动态约束配置已保存到: {args.output}")


if __name__ == '__main__':
    main() 
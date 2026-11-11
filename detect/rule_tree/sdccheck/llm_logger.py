"""LLM日志记录模块，用于记录所有LLM的输入和输出"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class LLMLogger:
    """LLM专用日志记录器，记录所有LLM交互的详细信息"""
    
    def __init__(self, log_dir: str = "logs", log_file: str = None):
        """
        初始化LLM日志记录器
        
        参数:
            log_dir: 日志目录
            log_file: 日志文件名（如果为None，则使用时间戳自动生成）
        """
        self.log_dir = Path(log_dir)
        
        # 如果没有指定日志文件名，则使用时间戳生成
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"llm_interactions_{timestamp}.log"
        
        self.log_file = self.log_dir / log_file
        
        # 确保日志目录存在
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置专门的LLM日志记录器
        self.logger = logging.getLogger("SDCCheck.LLMLogger")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建文件handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(file_handler)
            
            # 防止日志传播到根logger（避免重复输出）
            self.logger.propagate = False
    
    def log_llm_interaction(self, 
                           agent_name: str,
                           provider_name: str,
                           model_name: str,
                           system_prompt: Optional[str],
                           user_prompt: str,
                           response: str,
                           temperature: float,
                           max_tokens: int,
                           execution_time: float = None,
                           error: str = None) -> None:
        """
        记录LLM交互的详细信息
        
        参数:
            agent_name: Agent名称
            provider_name: LLM提供商名称
            model_name: 模型名称
            system_prompt: 系统提示
            user_prompt: 用户提示
            response: LLM响应
            temperature: 温度参数
            max_tokens: 最大token数
            execution_time: 执行时间（秒）
            error: 错误信息（如果有）
        """
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "provider_name": provider_name,
            "model_name": model_name,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "input": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "prompt_length": len(user_prompt) if user_prompt else 0,
                "system_prompt_length": len(system_prompt) if system_prompt else 0
            },
            "output": {
                "response": response,
                "response_length": len(response) if response else 0
            },
            "execution_time_seconds": execution_time,
            "error": error,
            "success": error is None
        }
        
        # 记录到日志文件
        log_message = f"LLM交互记录: {json.dumps(interaction_data, ensure_ascii=False, indent=2)}"
        
        if error:
            self.logger.error(log_message)
        else:
            self.logger.info(log_message)
    
    def log_summary(self, summary_data: Dict[str, Any]) -> None:
        """
        记录汇总信息
        
        参数:
            summary_data: 汇总数据
        """
        summary_message = f"LLM使用汇总: {json.dumps(summary_data, ensure_ascii=False, indent=2)}"
        self.logger.info(summary_message)


# 全局LLM日志记录器实例
_llm_logger_instance = None


def get_llm_logger(log_dir: str = "logs", log_file: str = None, force_new: bool = False) -> LLMLogger:
    """
    获取LLM日志记录器实例
    
    参数:
        log_dir: 日志目录
        log_file: 日志文件名（如果为None，则使用时间戳自动生成）
        force_new: 是否强制创建新实例（用于每次运行创建新日志文件）
        
    返回:
        LLMLogger: LLM日志记录器实例
    """
    global _llm_logger_instance
    
    # 如果强制创建新实例或者还没有实例，则创建新的
    if force_new or _llm_logger_instance is None:
        _llm_logger_instance = LLMLogger(log_dir, log_file)
    
    return _llm_logger_instance


def create_new_logger_session(log_dir: str = "logs") -> LLMLogger:
    """
    为新的运行会话创建新的日志记录器实例
    
    参数:
        log_dir: 日志目录
        
    返回:
        LLMLogger: 新的LLM日志记录器实例
    """
    return get_llm_logger(log_dir=log_dir, log_file=None, force_new=True)


def setup_unified_logging(log_dir: str = "logs") -> str:
    """
    设置统一的日志系统，让所有日志都写入到同一个带时间戳的日志文件中
    
    参数:
        log_dir: 日志目录
        
    返回:
        str: 日志文件路径
    """
    # 创建新的LLM日志会话
    llm_logger = create_new_logger_session(log_dir)
    log_file_path = str(llm_logger.log_file)
    
    # 配置根日志记录器，让所有应用日志也写入到同一个文件
    root_logger = logging.getLogger()
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建文件handler，写入到同一个日志文件
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler，用于显示重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到根logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    return log_file_path
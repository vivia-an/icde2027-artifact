"""
Token 消耗追踪器 - 用于记录 Agent SQL 生成过程中的 token 消耗

血缘位置: token_tracker.py
参照 gov 项目的 llm_config.py 中的 TokenTracker 实现
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class TokenTracker:
    """
    Token 消耗追踪器 - 单例模式
    用于追踪每次 SQL 生成过程中的 token 消耗
    
    使用方式:
        1. 调用 reset() 开始新一轮追踪
        2. LLM 调用后调用 add_usage() 累加消耗
        3. 调用 get_usage() 获取当前统计
        4. 调用 get_and_reset() 获取统计并重置
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logging.getLogger("SDCCheck.TokenTracker")
        self.reset()
        self.logger.info("[TokenTracker] 初始化 Token 追踪器单例")
    
    def reset(self):
        """重置 token 统计 - 每次 SQL 生成开始时调用"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.start_time = datetime.now()
        self.model_name = ""
        self.constraint_name = ""
        self.agent_name = ""
        self.logger.debug(f"[TokenTracker] 重置 token 统计，开始时间: {self.start_time.isoformat()}")
    
    def add_usage(self, usage: Dict[str, Any], model: str = "", agent_name: str = "", constraint_name: str = ""):
        """
        累加 token 使用量 - 每次 LLM 调用后调用
        
        参数:
            usage: token 使用信息字典，包含 prompt_tokens, completion_tokens, total_tokens
            model: 模型名称
            agent_name: Agent 名称
            constraint_name: 约束名称
        """
        if not usage:
            return
        
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total if total else (prompt + completion)
        self.call_count += 1
        
        if model:
            self.model_name = model
        if agent_name:
            self.agent_name = agent_name
        if constraint_name:
            self.constraint_name = constraint_name
        
        self.logger.debug(
            f"[TokenTracker] 累加 token: +{prompt}/{completion}/{total}, "
            f"累计: {self.prompt_tokens}/{self.completion_tokens}/{self.total_tokens}, "
            f"调用次数: {self.call_count}"
        )
    
    def get_usage(self) -> Dict[str, Any]:
        """获取当前的 token 使用统计"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        result = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "model": self.model_name,
            "agent_name": self.agent_name,
            "constraint_name": self.constraint_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2)
        }
        
        self.logger.debug(f"[TokenTracker] 获取统计: {result}")
        return result
    
    def get_and_reset(self) -> Dict[str, Any]:
        """获取统计并重置 - 用于一次 SQL 生成结束时"""
        result = self.get_usage()
        self.reset()
        return result
    
    def set_context(self, constraint_name: str = "", agent_name: str = ""):
        """设置当前上下文（约束名称、Agent 名称）"""
        if constraint_name:
            self.constraint_name = constraint_name
        if agent_name:
            self.agent_name = agent_name


# 全局 TokenTracker 实例
_token_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """获取全局 TokenTracker 实例"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker


class SQLTokenConsumptionManager:
    """
    SQL Token 消耗管理器
    负责将 token 消耗记录保存到 JSON 文件中
    
    血缘位置: token_tracker.py
    """
    
    def __init__(self, output_file: Optional[str] = None):
        """
        初始化管理器
        
        参数:
            output_file: 输出文件路径，默认为 config/sql_token_consumption.json
        """
        self.logger = logging.getLogger("SDCCheck.SQLTokenConsumptionManager")
        
        if output_file is None:
            # 默认输出到 config 目录下
            current_dir = Path(__file__).parent
            output_file = str(current_dir.parent / "config" / "sql_token_consumption.json")
        
        self.output_file = Path(output_file)
        self.consumption_records: Dict[str, Dict[str, Any]] = {}
        
        # 加载已有记录
        self._load_existing_records()
    
    def _load_existing_records(self):
        """加载已有的消耗记录"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.consumption_records = data.get("sql_token_consumption", {})
                    self.logger.info(f"[SQLTokenConsumptionManager] 加载已有记录: {len(self.consumption_records)} 条")
            except Exception as e:
                self.logger.warning(f"[SQLTokenConsumptionManager] 加载记录失败: {e}")
                self.consumption_records = {}
        else:
            self.consumption_records = {}
    
    def record_consumption(self, constraint_name: str, token_usage: Dict[str, Any], 
                          sql: str = "", analysis_summary: str = ""):
        """
        记录约束验证的 token 消耗
        
        参数:
            constraint_name: 约束名称
            token_usage: token 使用统计
            sql: 生成的 SQL 语句
            analysis_summary: 分析摘要
        """
        record = {
            "constraint_name": constraint_name,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "call_count": token_usage.get("call_count", 0),
            "model": token_usage.get("model", ""),
            "agent_name": token_usage.get("agent_name", ""),
            "generated_sql": sql[:500] if sql else "",  # 只保存前500字符
            "analysis_summary": analysis_summary[:200] if analysis_summary else "",
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": token_usage.get("duration_seconds", 0)
        }
        
        # 使用约束名称作为 key
        self.consumption_records[constraint_name] = record
        
        self.logger.info(
            f"[SQLTokenConsumptionManager] 记录消耗 - {constraint_name}: "
            f"total_tokens={record['total_tokens']}, call_count={record['call_count']}"
        )
        
        # 保存到文件
        self._save_records()
    
    def _save_records(self):
        """保存消耗记录到文件"""
        try:
            # 确保目录存在
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 计算汇总统计
            total_prompt = sum(r.get("prompt_tokens", 0) for r in self.consumption_records.values())
            total_completion = sum(r.get("completion_tokens", 0) for r in self.consumption_records.values())
            total_tokens = sum(r.get("total_tokens", 0) for r in self.consumption_records.values())
            total_calls = sum(r.get("call_count", 0) for r in self.consumption_records.values())
            
            output_data = {
                "metadata": {
                    "description": "SDCCheck Agent SQL 生成 Token 消耗记录",
                    "last_updated": datetime.now().isoformat(),
                    "total_constraints": len(self.consumption_records),
                    "summary": {
                        "total_prompt_tokens": total_prompt,
                        "total_completion_tokens": total_completion,
                        "total_tokens": total_tokens,
                        "total_llm_calls": total_calls
                    }
                },
                "sql_token_consumption": self.consumption_records
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"[SQLTokenConsumptionManager] 记录已保存到: {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"[SQLTokenConsumptionManager] 保存记录失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取消耗汇总"""
        total_prompt = sum(r.get("prompt_tokens", 0) for r in self.consumption_records.values())
        total_completion = sum(r.get("completion_tokens", 0) for r in self.consumption_records.values())
        total_tokens = sum(r.get("total_tokens", 0) for r in self.consumption_records.values())
        total_calls = sum(r.get("call_count", 0) for r in self.consumption_records.values())
        
        return {
            "total_constraints_verified": len(self.consumption_records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_llm_calls": total_calls,
            "average_tokens_per_constraint": round(total_tokens / len(self.consumption_records), 2) if self.consumption_records else 0
        }
    
    def print_summary(self) -> str:
        """打印消耗汇总信息"""
        summary = self.get_summary()
        
        output = f"""
╔════════════════════════════════════════════════════════════════╗
║              SDCCheck SQL 生成 Token 消耗汇总                  ║
╠════════════════════════════════════════════════════════════════╣
║  验证约束数量:     {summary['total_constraints_verified']:>10}                          ║
║  总 Prompt Tokens: {summary['total_prompt_tokens']:>10}                          ║
║  总 Completion Tokens: {summary['total_completion_tokens']:>10}                      ║
║  总 Token 消耗:    {summary['total_tokens']:>10}                          ║
║  总 LLM 调用次数:  {summary['total_llm_calls']:>10}                          ║
║  平均每约束 Token: {summary['average_tokens_per_constraint']:>10.2f}                          ║
╚════════════════════════════════════════════════════════════════╝
"""
        return output


# 全局 SQLTokenConsumptionManager 实例
_consumption_manager: Optional[SQLTokenConsumptionManager] = None


def get_consumption_manager(output_file: Optional[str] = None) -> SQLTokenConsumptionManager:
    """获取全局 SQLTokenConsumptionManager 实例"""
    global _consumption_manager
    if _consumption_manager is None:
        _consumption_manager = SQLTokenConsumptionManager(output_file)
    return _consumption_manager



















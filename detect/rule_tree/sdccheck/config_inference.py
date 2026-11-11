#!/usr/bin/env python3
"""从数据库数据推断训练配置的模块"""

import duckdb
from typing import Optional
from pathlib import Path

from .models import TrainingConfig


class TrainingConfigInference:
    """从数据库数据推断训练配置的类"""
    
    def __init__(self):
        self.conn = None
        
    def infer_from_database(self, db_path: str) -> TrainingConfig:
        """
        从数据库数据推断训练配置
        
        参数:
            db_path: 数据库文件路径
            
        返回:
            TrainingConfig: 推断出的训练配置
        """
        # 确保数据库文件存在
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
        # 连接数据库（只读模式）
        self.conn = duckdb.connect(str(db_path), read_only=True)
        
        try:
            # 推断各种并行配置
            config = self._analyze_parallel_config()
            return config
        finally:
            # 关闭连接
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def _analyze_parallel_config(self) -> TrainingConfig:
        """分析并行配置"""
        # 获取所有并行配置的唯一值
        tp_values = self._get_unique_values('tp')
        dp_values = self._get_unique_values('dp')
        pp_values = self._get_unique_values('pp')
        ep_values = self._get_unique_values('ep')
        cp_values = self._get_unique_values('cp')
        etp_values = self._get_unique_values('etp')
        
        # 计算并行度（最大值 + 1，因为rank从0开始）
        tp = max(tp_values) + 1 if tp_values else 1
        dp = max(dp_values) + 1 if dp_values else 1
        pp = max(pp_values) + 1 if pp_values else 1
        ep = max(ep_values) + 1 if ep_values else 1
        sp = max(cp_values) + 1 if cp_values else 1  # cp对应sequence parallel
        
        
        return TrainingConfig(
            dp=dp,
            tp=tp,
            pp=pp,
            ep=ep,
            sp=sp
        )
    
    def _get_unique_values(self, field: str) -> list:
        """获取指定字段的唯一值"""
        query = f"""
        SELECT DISTINCT CAST(JSON_EXTRACT(data, '$.{field}') AS INTEGER) as {field}
        FROM coredump 
        WHERE JSON_EXTRACT(data, '$.{field}') IS NOT NULL
        ORDER BY {field}
        """
        
        try:
            result = self.conn.execute(query).fetchall()
            return [row[0] for row in result if row[0] is not None]
        except Exception:
            return []
    

    def get_database_summary(self, db_path: str) -> dict:
        """获取数据库的基本统计信息"""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
        conn = duckdb.connect(str(db_path), read_only=True)
        
        try:
            # 基本统计
            basic_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT step) as unique_steps,
                COUNT(DISTINCT stage) as unique_stages,
                MIN(step) as min_step,
                MAX(step) as max_step
            FROM coredump
            """).fetchone()
            
            # 参数统计
            param_stats = conn.execute("""
            SELECT 
                COUNT(DISTINCT JSON_EXTRACT(data, '$.name')) as unique_params
            FROM coredump
            """).fetchone()
            
            return {
                'total_records': basic_stats[0],
                'unique_steps': basic_stats[1],
                'unique_stages': basic_stats[2],
                'min_step': basic_stats[3],
                'max_step': basic_stats[4],
                'unique_params': param_stats[0]
            }
        finally:
            conn.close()
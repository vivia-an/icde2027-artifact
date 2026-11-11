import duckdb
import pandas as pd
import logging
from typing import Any, Optional, List, Dict, Tuple
from pathlib import Path
import threading


class DatabaseExecutor:
    """数据库执行器，负责执行SQL查询并返回结果（支持并发查询）"""
    
    def __init__(self):
        self.conn = None
        self.db_path = None
        self._lock = threading.Lock()  # 线程锁，保护连接操作
        self.logger = logging.getLogger("SDCCheck.DatabaseExecutor")
        
    def connect(self, db_path: str) -> None:
        """
        连接到数据库
        
        参数:
            db_path: 数据库文件路径
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
        with self._lock:
            self.db_path = str(db_path)
            # DuckDB支持多线程并发读取
            self.conn = duckdb.connect(self.db_path, read_only=True)
            self.logger.info(f"已连接到数据库: {db_path}")
    
    def execute(self, sql: str) -> pd.DataFrame:
        """
        执行SQL查询并返回结果（线程安全）
        
        参数:
            sql: SQL查询语句
            
        返回:
            pd.DataFrame: 查询结果
        """
        if self.conn is None:
            raise RuntimeError("尚未连接到数据库，请先调用connect()方法")
            
        try:
            # DuckDB连接本身是线程安全的，可以并发执行查询
            result = self.conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            # 记录错误并重新抛出
            self.logger.error(f"执行SQL时出错: {e}")
            self.logger.debug(f"SQL: {sql}")
            raise
    
    def execute_batch(self, sql_list: List[str]) -> List[pd.DataFrame]:
        """
        批量执行SQL查询（顺序执行）
        
        参数:
            sql_list: SQL查询语句列表
            
        返回:
            List[pd.DataFrame]: 查询结果列表
        """
        results = []
        
        for sql in sql_list:
            try:
                result = self.execute(sql)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量执行SQL失败: {e}")
                # 添加空DataFrame作为错误占位
                results.append(pd.DataFrame())
        
        return results
    
    def execute_with_metadata(self, sql: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行SQL查询并返回结果和元数据
        
        参数:
            sql: SQL查询语句
            
        返回:
            Tuple[pd.DataFrame, Dict]: (查询结果, 元数据)
        """
        if self.conn is None:
            raise RuntimeError("尚未连接到数据库，请先调用connect()方法")
        
        try:
            result = self.conn.execute(sql)
            df = result.fetchdf()
            
            # 构建元数据
            metadata = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"执行SQL时出错: {e}")
            raise
    
    def is_connected(self) -> bool:
        """检查是否已连接到数据库"""
        return self.conn is not None
    
    def close(self) -> None:
        """关闭数据库连接"""
        with self._lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
                self.logger.info("数据库连接已关闭") 
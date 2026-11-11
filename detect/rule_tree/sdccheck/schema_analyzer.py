import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from .models import SchemaInfo


class SchemaAnalyzer:
    """数据库模式分析器，负责分析数据库结构和JSON字段"""
    
    def __init__(self):
        self.conn = None
        
    def analyze(self, db_path: str) -> SchemaInfo:
        """
        分析数据库模式并返回SchemaInfo对象
        
        参数:
            db_path: 数据库文件路径
            
        返回:
            SchemaInfo: 包含数据库模式信息的对象
        """
        # 确保数据库文件存在
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
        # 连接数据库
        self.conn = duckdb.connect(str(db_path), read_only=True)
        
        # 获取所有表信息
        tables_info = {}
        tables = self._get_tables()
        
        for table in tables:
            table_info = {
                "columns": self._get_columns(table),
                "json_keys": {},
            }
            
            # 查找JSON字段并提取键
            for column in table_info["columns"]:
                if column["type"].lower() in ("json", "varchar"):  # DuckDB中JSON可能存储为VARCHAR
                    try:
                        json_keys = self._extract_json_keys(table, column["name"])
                        if json_keys:
                            table_info["json_keys"][column["name"]] = json_keys
                    except Exception as e:
                        # 如果该字段不是有效的JSON，则跳过
                        continue
                        
            tables_info[table] = table_info
            
        # 关闭连接
        self.conn.close()
        self.conn = None
        
        return SchemaInfo(tables=tables_info)
    
    def _get_tables(self) -> List[str]:
        """获取数据库中的所有表名"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        """
        result = self.conn.execute(query).fetchall()
        return [row[0] for row in result]
    
    def _get_columns(self, table_name: str) -> List[Dict[str, str]]:
        """获取指定表的所有列信息"""
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'main' AND table_name = '{table_name}'
        """
        result = self.conn.execute(query).fetchall()
        return [{"name": row[0], "type": row[1]} for row in result]
    
    def _extract_json_keys(self, table_name: str, column_name: str, limit: int = 100) -> List[str]:
        """提取JSON字段中的键"""
        try:
            # 获取一条示例数据并解析JSON
            query = f"""
            SELECT {column_name} 
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            LIMIT 1
            """
            result = self.conn.execute(query).fetchall()
            if result:
                json_str = result[0][0]
                # 尝试使用DuckDB的JSON函数解析
                try:
                    keys_query = f"SELECT json_keys('{json_str}')"
                    keys_result = self.conn.execute(keys_query).fetchall()
                    if keys_result and keys_result[0][0]:
                        return keys_result[0][0]
                except:
                    # 如果DuckDB JSON函数失败，尝试Python解析
                    import json
                    try:
                        json_data = json.loads(json_str)
                        if isinstance(json_data, dict):
                            return list(json_data.keys())
                    except:
                        pass
        except Exception as e:
            pass
        
        return []
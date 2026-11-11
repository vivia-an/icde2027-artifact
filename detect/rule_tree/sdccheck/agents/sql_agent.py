from typing import Dict, Any
import json

from .base_agent import BaseAgent, LLMConfig
from ..models import Constraint, ConstraintType


class SQLAgent(BaseAgent):
    """SQL生成Agent，使用LLM智能生成SQL查询语句"""
    
    def __init__(self, config: LLMConfig, provider_name: str = "mock"):
        super().__init__(config, "SQLAgent", provider_name)
    
    def process(self, input_data: Any) -> str:
        """处理输入数据并生成SQL"""
        constraint = input_data
        return self.generate_sql(constraint)
    
    def generate_sql(self, constraint: Constraint) -> str:
        """
        使用LLM根据约束对象生成SQL查询语句
        
        参数:
            constraint: 约束对象
            
        返回:
            str: 生成的SQL查询语句
        """
        self.logger.info(f"开始为约束[{constraint.name}]生成SQL")
        
        # 设置 token 追踪上下文
        self.token_tracker.set_context(constraint_name=constraint.name, agent_name=self.name)
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 构建用户提示
        user_prompt = self._build_user_prompt(constraint)
        # 调用LLM（传递约束名称用于追踪）
        response = self._call_llm(user_prompt, system_prompt, constraint_name=constraint.name)
        
        # 解析响应
        try:
            parsed_response = self._parse_json_response(response)
            sql = parsed_response.get("sql", "")
            
            if not sql:
                raise ValueError("LLM未返回有效的SQL语句")
            
            self.logger.info(f"成功生成SQL，长度: {len(sql)}")
            self.logger.debug(f"生成的SQL: {sql}")
            
            return sql
            
        except Exception as e:
            self.logger.error(f"解析SQL生成结果失败: {e}")
            return ""
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """
你是一个专业的SQL查询生成专家，专门为分布式训练数据检查生成SQL查询。 当前使用 数据库为duckdb。你是duckdb的专家，熟悉duckdb的语法和函数。同时你还非常熟悉Megatron的训练流程和数据结构。知道里边约束的含义，当前需要根据megatron约束来依据数据结构生成SQL查询，当前dump的数据校验主要使用checksum将权重/参数进行hash，然后进行比较。不一致的比较也是比较不同分片约束的权重/参数。尽量生成简单的统计，进行对比 应该在同一步骤 同一阶段下的 不同分片的 权重进行对比，tp,dp,pp 分别表示 对应tensor并行的不同分片 如0，1，dp就是data 并行的不同分片表示 如 0，1，所以可以采用如下格式进行异常统计，所有权重的哈希全部都在 cksum 中，没有其他字段存放hash值：dp 等是分片标识，如果你做不一致统计，就不要筛选这个，不要有这种 
	AND CAST(json_extract_string(data, '$.dp') AS INTEGER) > 1 这种筛选，直接全部统计出来。
···
WITH param_stats AS (
        SELECT 
            json_extract_string(data, '$.name') as param_name,
            CAST(json_extract_string(data, '$.dp') AS INTEGER) as dp_rank,
            json_extract_string(data, '$.cksum') as cksum,
            step,
            stage
        FROM coredump
        WHERE stage = 'model-after-optimizer-step'
          AND json_extract_string(data, '$.cksum') IS NOT NULL
    ),
    inconsistency_check AS (
        SELECT 
            param_name,
            step,
            COUNT(DISTINCT cksum) as distinct_cksums,
            COUNT(*) as total_records,
            array_agg(DISTINCT cksum) as all_cksums,
            array_agg(DISTINCT dp_rank) as dp_ranks
        FROM param_stats
        GROUP BY param_name, step
        HAVING COUNT(DISTINCT cksum) > 1
    )
    SELECT * FROM inconsistency_check
    ORDER BY step, param_name
 ···   

数据库模式信息：
- 表名：coredump
- 列结构：
  * step (INTEGER): 训练步骤
  * stage (VARCHAR): 训练阶段
  * data (JSON): 包含所有详细信息的JSON字段

JSON字段(data)包含的信息：
- name: 权重/参数名称
- tp: 张量并行rank  
- dp: 数据并行rank  如果dp>1 ,那么对应值就是 0,1 ...
- pp: 管道并行rank
- cksum: 校验和    
- size: 数据大小  
- valid: 是否有效
data字段的JSON中数据示例：

{"name": "module.module.embedding.word_embeddings.weight", "cksum": "70ce92b29ec4b320b46464c71b84d15b1a1c636b715244cc9b79520d2a356c42", "shape": [50304, 128], "type": "torch.cuda.BFloat16Tensor", "requires_grad": true, "grad_cksum": null, "grad_shape": null, "grad_type": null, "tp": 0, "dp": 0, "cp": 0, "pp": 0, "ep": 0, "etp": 0}
    {"name": "module.module.decoder.layers.0.self_attention.linear_proj.weight", "cksum": "5106605952167753fabb1ef3808cb1174ac3d6d088d4eb19ab69254761e7e73d", "shape": [128, 128], "type": "torch.cuda.BFloat16Tensor", "requires_grad": true, "grad_cksum": null, "grad_shape": null, "grad_type": null, "tp": 0, "dp": 0, "cp": 0, "pp": 0, "ep": 0, "etp": 0}


**重要提醒：**
- 表coredump只有三个直接列：step, stage, data
- 所有其他信息（name, tp, dp, pp, cksum等）都存储在data这个JSON字段中
- 访问JSON字段必须使用：JSON_EXTRACT(data, '$.字段名')
- 绝对不能直接使用 table.name, table.dp, table.cksum 等，这些列不存在！
- 正确示例：JSON_EXTRACT(data, '$.dp') AS dp
- 错误示例：d.dp（这会导致"列不存在"错误）

你需要理解给出的约束并生成相应的SQL！！！
生成的SQL应该：
- 高效且准确
- 能够识别违反约束的记录
- 提供足够的信息用于后续分析
- 使用标准SQL语法（兼容DuckDB）
- 正确使用JSON_EXTRACT函数访问data字段中的信息
- 绝对不能直接访问不存在的列名

根据示例数据分析 如何查出约束冲突的异常统计出来
函数约束检查，duckdb常用函数说明：DuckDB JSON函数完整列表（仅支持以下函数）：

1. JSON_EXTRACT(data, '$.field') 
   - 返回JSON值，自动识别类型
   - 适用于：数字、布尔值、NULL
   - 示例：JSON_EXTRACT(data, '$.dp') = 1
   - 示例：JSON_EXTRACT(data, '$.requires_grad') = true

2. JSON_EXTRACT_STRING(data, '$.field')
   - 强制返回字符串类型
   - 适用于：字符串字段，如 name, cksum, stage 等
   - 示例：JSON_EXTRACT_STRING(data, '$.name') = 'param'

3. 不存在的函数（禁止使用）：
    JSON_EXTRACT_BOOLEAN - 没有这个函数！
    JSON_EXTRACT_INT - 没有这个函数！
    JSON_EXTRACT_NUMBER - 没有这个函数！
   
布尔值正确用法：
 JSON_EXTRACT(data, '$.requires_grad') = true
 JSON_EXTRACT(data, '$.requires_grad') = false
 JSON_EXTRACT_BOOLEAN(data, '$.requires_grad') -- 不存在！

        """
    
    def _build_user_prompt(self, constraint: Constraint) -> str:
        """构建用户提示"""
        constraint_str = self._format_data_for_prompt(constraint.dict())
        
        return f"""
请为以下约束生成相应的SQL查询语句：

约束信息：
{constraint_str}

要求：
1. 生成的SQL应该能够检测违反该约束的情况
2. 如果没有违规，查询应该返回空结果
3. 如果有违规，查询应该返回违规的记录或统计信息
4. SQL应该高效且准确
5. 使用标准SQL语法，满足DuckDB的语法限制

特别注意：
- 对于一致性约束：查找要求相同或不同的记录，但是实际与要求相反的记录
- 对于分区约束：查找应该不同但实际相同的记录
- 对于完整性约束：统计实际数量与期望数量的差异
- 对于有效性约束：查找无效或异常的记录

请以JSON格式返回结果：
{{
    "sql": "生成的SQL查询语句",
    "explanation": "SQL的作用和逻辑说明",
    "expected_result": "期望的查询结果描述"
}}

** JSON格式严格要求（必须遵守，否则解析失败！）：**

1. **SQL字段必须是单行字符串**
   -  正确：{{"sql": "SELECT * FROM table WHERE condition"}}
   -  错误：{{"sql": "SELECT *
                     FROM table"}}  (包含换行符)

2. **不要在JSON字符串中使用任何控制字符**
   - 禁止：换行符 (\\n)、制表符 (\\t)、回车符 (\\r)
   - 如果SQL很长，直接写成一行，用空格分隔各部分
   
3. **不要使用反斜杠进行续行**
   -  错误：{{"sql": "SELECT * \\
                     FROM table"}}
   -  正确：{{"sql": "SELECT * FROM table"}}

4. **确保JSON格式完整有效**
   - 所有字符串必须用双引号
   - 所有字段必须正确闭合
   - 不要有多余的逗号或括号

**示例对比：**

错误的JSON（会导致解析失败）：
```
{{
    "sql": "SELECT name, 
            cksum 
            FROM coredump",  ←  包含换行
    "explanation": "查询..."
}}
```

正确的JSON（能成功解析）：
```
{{
    "sql": "SELECT name, cksum FROM coredump",  ←  单行
    "explanation": "查询参数名和校验和"
}}

```

**请严格按照以上格式返回JSON，确保sql字段是连续的单行字符串！**
        """
    

    def validate_sql(self, sql: str) -> bool:
        """
        验证生成的SQL语法是否正确
        
        参数:
            sql: 要验证的SQL语句
            
        返回:
            bool: SQL是否有效
        """
        try:
            # 这里可以添加SQL语法验证逻辑
            # 例如使用sqlparse库或者简单的关键字检查
            
            # 基本的关键字检查
            sql_upper = sql.upper().strip()
            
            # 检查是否包含基本的SQL关键字
            if not sql_upper.startswith('SELECT'):
                return False
            
            # 检查是否包含FROM子句
            if 'FROM' not in sql_upper:
                return False
            
            # 检查是否有明显的语法错误
            if sql.count('(') != sql.count(')'):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证SQL时出错: {e}")
            return False
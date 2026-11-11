"""并发执行器，支持批量并发执行SQL查询"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Optional, Callable
from .models import ConstraintNode, BatchExecutionResult, AnalysisReport
from .database_executor import DatabaseExecutor
from .generators.llm_sql_generator import LLMSQLGenerator
from .generators.llm_result_analyzer import LLMResultAnalyzer
import pandas as pd


class ConcurrentConstraintExecutor:
    """并发约束执行器，支持批量并发执行约束检查"""
    
    def __init__(self, 
                 db_executor: DatabaseExecutor,
                 sql_generator: LLMSQLGenerator,
                 result_analyzer: LLMResultAnalyzer,
                 max_workers: int = 10):
        """
        初始化并发执行器
        
        参数:
            db_executor: 数据库执行器
            sql_generator: SQL生成器
            result_analyzer: 结果分析器
            max_workers: 最大并发数（默认10）
        """
        self.db_executor = db_executor
        self.sql_generator = sql_generator
        self.result_analyzer = result_analyzer
        self.max_workers = max_workers
        self.logger = logging.getLogger("SDCCheck.ConcurrentExecutor")
        
        self.logger.info(f"初始化并发执行器，最大并发数: {max_workers}")
    
    def execute_batch(self, nodes: List[ConstraintNode], batch_id: int = 0) -> BatchExecutionResult:
        """
        并发执行一批约束检查
        
        参数:
            nodes: 要执行的节点列表
            batch_id: 批次ID
            
        返回:
            BatchExecutionResult: 批次执行结果
        """
        if not nodes:
            return BatchExecutionResult(
                batch_id=batch_id,
                node_ids=[],
                reports=[],
                execution_time=0.0
            )
        
        self.logger.info(f"开始执行批次 {batch_id}，共 {len(nodes)} 个约束")
        start_time = time.time()
        
        # 准备结果容器
        reports: List[AnalysisReport] = []
        node_ids = [node.node_id for node in nodes]
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_node: Dict[Future, ConstraintNode] = {}
            
            for node in nodes:
                future = executor.submit(self._execute_single_constraint, node)
                future_to_node[future] = node
            
            # 收集结果
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                
                try:
                    report = future.result()
                    reports.append(report)
                    self.logger.debug(f"节点 {node.node_id} 执行完成: {report.status}")
                    
                except Exception as e:
                    self.logger.error(f"节点 {node.node_id} 执行失败: {e}")
                    # 创建错误报告
                    error_report = AnalysisReport(
                        constraint=node.constraint,
                        status="error",
                        summary=f"执行过程中出错: {str(e)}",
                        violations=[],
                        raw_data=None,
                        node_id=node.node_id
                    )
                    reports.append(error_report)
        
        # 计算统计信息
        execution_time = time.time() - start_time
        success_count = sum(1 for r in reports if r.status == "pass")
        failure_count = sum(1 for r in reports if r.status == "fail")
        error_count = sum(1 for r in reports if r.status == "error")
        
        result = BatchExecutionResult(
            batch_id=batch_id,
            node_ids=node_ids,
            reports=reports,
            execution_time=execution_time,
            success_count=success_count,
            failure_count=failure_count,
            error_count=error_count
        )
        
        self.logger.info(
            f"批次 {batch_id} 执行完成，"
            f"耗时 {execution_time:.2f}s，"
            f"成功: {success_count}，失败: {failure_count}，错误: {error_count}"
        )
        
        return result
    
    def _execute_single_constraint(self, node: ConstraintNode) -> AnalysisReport:
        """
        执行单个约束检查
        
        参数:
            node: 约束节点
            
        返回:
            AnalysisReport: 分析报告
        """
        constraint = node.constraint
        exec_start = time.time()
        
        try:
            # 如果是虚拟节点，跳过执行
            if constraint.params.get("virtual_node"):
                self.logger.debug(f"跳过虚拟节点: {constraint.name}")
                return AnalysisReport(
                    constraint=constraint,
                    status="pass",
                    summary="虚拟节点，无需检查",
                    violations=[],
                    raw_data=None,
                    execution_time=0.0,
                    node_id=node.node_id
                )
            
            # 生成SQL
            self.logger.debug(f"为约束 '{constraint.name}' 生成SQL")
            sql = self.sql_generator.generate(constraint)
            
            # 执行SQL
            self.logger.debug(f"执行SQL查询: {constraint.name}")
            result = self.db_executor.execute(sql)
            
            # 分析结果
            self.logger.debug(f"分析查询结果: {constraint.name}")
            report = self.result_analyzer.analyze(constraint, result)
            
            # 设置执行时间和节点ID
            report.execution_time = time.time() - exec_start
            report.node_id = node.node_id
            
            return report
            
        except Exception as e:
            self.logger.error(f"执行约束 '{constraint.name}' 时出错: {e}")
            
            # 返回错误报告
            return AnalysisReport(
                constraint=constraint,
                status="error",
                summary=f"执行失败: {str(e)}",
                violations=[],
                raw_data=None,
                execution_time=time.time() - exec_start,
                node_id=node.node_id
            )
    
    def execute_level_batch(self, nodes: List[ConstraintNode], 
                           batch_size: int = 10) -> List[BatchExecutionResult]:
        """
        分批次执行同一层级的所有节点
        
        参数:
            nodes: 节点列表
            batch_size: 每批大小（默认10）
            
        返回:
            List[BatchExecutionResult]: 所有批次的执行结果
        """
        if not nodes:
            return []
        
        self.logger.info(f"开始分批执行 {len(nodes)} 个节点，每批 {batch_size} 个")
        
        batch_results = []
        
        # 按batch_size分批
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_id = i // batch_size
            
            result = self.execute_batch(batch_nodes, batch_id)
            batch_results.append(result)
        
        total_time = sum(r.execution_time for r in batch_results)
        total_success = sum(r.success_count for r in batch_results)
        total_failure = sum(r.failure_count for r in batch_results)
        total_error = sum(r.error_count for r in batch_results)
        
        self.logger.info(
            f"分批执行完成，共 {len(batch_results)} 批，"
            f"总耗时 {total_time:.2f}s，"
            f"成功: {total_success}，失败: {total_failure}，错误: {total_error}"
        )
        
        return batch_results
    
    def execute_tree_level_by_level(self, 
                                    tree_manager,
                                    start_level: int = 0,
                                    stop_on_all_pass: bool = False) -> Dict[str, Any]:
        """
        逐层执行约束树，失败节点的子节点会被继续执行
        
        参数:
            tree_manager: 约束树管理器
            start_level: 起始层级（默认从0开始）
            stop_on_all_pass: 如果某一层全部通过，是否停止（默认False）
            
        返回:
            Dict: 执行统计信息
        """
        from .constraint_tree import ConstraintTreeManager
        
        if not isinstance(tree_manager, ConstraintTreeManager):
            raise TypeError("tree_manager必须是ConstraintTreeManager实例")
        
        if not tree_manager.tree:
            self.logger.warning("约束树为空")
            return {}
        
        self.logger.info(f"开始逐层执行约束树，起始层级: {start_level}")
        
        # 统计信息
        stats = {
            "total_nodes_executed": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_errors": 0,
            "levels_executed": [],
            "total_time": 0.0
        }
        
        # 获取要执行的节点（起始层级）
        current_level = start_level
        nodes_to_execute = tree_manager.get_nodes_by_level(current_level)
        
        while nodes_to_execute:
            self.logger.info(f"执行层级 {current_level}，共 {len(nodes_to_execute)} 个节点")
            
            # 执行当前层级
            level_start = time.time()
            batch_results = self.execute_level_batch(nodes_to_execute, batch_size=self.max_workers)
            level_time = time.time() - level_start
            
            # 更新节点报告
            for batch_result in batch_results:
                for report in batch_result.reports:
                    if report.node_id:
                        tree_manager.update_node_report(report.node_id, report)
            
            # 统计本层结果
            level_passed = sum(r.success_count for r in batch_results)
            level_failed = sum(r.failure_count for r in batch_results)
            level_errors = sum(r.error_count for r in batch_results)
            
            stats["total_nodes_executed"] += len(nodes_to_execute)
            stats["total_passed"] += level_passed
            stats["total_failed"] += level_failed
            stats["total_errors"] += level_errors
            stats["total_time"] += level_time
            stats["levels_executed"].append({
                "level": current_level,
                "nodes_count": len(nodes_to_execute),
                "passed": level_passed,
                "failed": level_failed,
                "errors": level_errors,
                "time": level_time
            })
            
            self.logger.info(
                f"层级 {current_level} 执行完成: "
                f"通过 {level_passed}，失败 {level_failed}，错误 {level_errors}"
            )
            
            # 检查是否停止
            if stop_on_all_pass and level_failed == 0 and level_errors == 0:
                self.logger.info(f"层级 {current_level} 全部通过，停止执行")
                break
            
            # 获取下一层要执行的节点（失败节点的子节点）
            next_nodes = tree_manager.get_unchecked_children_of_failed_nodes()
            
            if not next_nodes:
                self.logger.info("没有更多未检查的子节点，执行完成")
                break
            
            nodes_to_execute = next_nodes
            current_level += 1
        
        self.logger.info(
            f"约束树执行完成，"
            f"共执行 {stats['total_nodes_executed']} 个节点，"
            f"总耗时 {stats['total_time']:.2f}s"
        )
        
        return stats
    
    def execute_nodes_with_callback(self,
                                   nodes: List[ConstraintNode],
                                   callback: Optional[Callable[[AnalysisReport], None]] = None) -> List[AnalysisReport]:
        """
        执行节点并在每个节点完成时调用回调函数
        
        参数:
            nodes: 节点列表
            callback: 回调函数，接收AnalysisReport参数
            
        返回:
            List[AnalysisReport]: 所有报告
        """
        reports = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_node = {
                executor.submit(self._execute_single_constraint, node): node
                for node in nodes
            }
            
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                
                try:
                    report = future.result()
                    reports.append(report)
                    
                    # 调用回调
                    if callback:
                        callback(report)
                        
                except Exception as e:
                    self.logger.error(f"节点 {node.node_id} 执行失败: {e}")
                    error_report = AnalysisReport(
                        constraint=node.constraint,
                        status="error",
                        summary=f"执行失败: {str(e)}",
                        violations=[],
                        raw_data=None,
                        node_id=node.node_id
                    )
                    reports.append(error_report)
                    
                    if callback:
                        callback(error_report)
        
        return reports









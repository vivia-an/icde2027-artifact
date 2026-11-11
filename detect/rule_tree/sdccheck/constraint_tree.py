"""约束树模块，管理约束的层级关系和树结构"""

import logging
from typing import List, Dict, Optional, Set
from .models import Constraint, ConstraintNode, ConstraintTree, AnalysisReport
import uuid


class ConstraintTreeManager:
    """约束树管理器，负责构建和管理约束树"""
    
    def __init__(self):
        self.logger = logging.getLogger("SDCCheck.ConstraintTreeManager")
        self.tree: Optional[ConstraintTree] = None
    
    def build_tree_from_constraints(self, constraints: List[Constraint]) -> ConstraintTree:
        """
        从约束列表构建约束树
        
        参数:
            constraints: 约束列表（已经包含parent/children信息）
            
        返回:
            ConstraintTree: 构建的约束树
        """
        self.logger.info(f"开始构建约束树，共{len(constraints)}个约束")
        
        # 创建节点映射（约束名称 -> 节点）
        name_to_node: Dict[str, ConstraintNode] = {}
        
        # 第一遍：创建所有节点
        for constraint in constraints:
            node_id = self._generate_node_id(constraint.name)
            node = ConstraintNode(
                node_id=node_id,
                constraint=constraint,
                level=constraint.level,
                parent_id=None  # 稍后设置
            )
            name_to_node[constraint.name] = node
        
        # 第二遍：建立父子关系
        root_nodes = []
        for constraint in constraints:
            node = name_to_node[constraint.name]
            
            # 设置父节点
            if constraint.parent and constraint.parent in name_to_node:
                parent_node = name_to_node[constraint.parent]
                node.parent_id = parent_node.node_id
                parent_node.children.append(node)
                self.logger.debug(f"建立关系: {constraint.name} -> 父节点: {constraint.parent}")
            else:
                # 根节点
                root_nodes.append(node)
                self.logger.debug(f"根节点: {constraint.name}")
            
            # 设置子节点（验证children字段）
            for child_name in constraint.children:
                if child_name not in name_to_node:
                    self.logger.warning(f"约束 '{constraint.name}' 引用的子节点 '{child_name}' 不存在")
        
        # 创建树对象
        node_map = {node.node_id: node for node in name_to_node.values()}
        max_depth = self._calculate_max_depth(root_nodes)
        
        tree = ConstraintTree(
            root_nodes=root_nodes,
            node_map=node_map,
            max_depth=max_depth,
            total_nodes=len(constraints)
        )
        
        self.tree = tree
        self.logger.info(f"约束树构建完成: {len(root_nodes)}个根节点, 最大深度{max_depth}, 总计{len(constraints)}个节点")
        
        return tree
    
    def _generate_node_id(self, constraint_name: str) -> str:
        """生成节点唯一ID"""
        # 使用约束名称的hash + 短UUID确保唯一性
        short_uuid = str(uuid.uuid4())[:8]
        # 清理约束名称，只保留字母数字和下划线
        clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in constraint_name)
        return f"{clean_name}_{short_uuid}"
    
    def _calculate_max_depth(self, root_nodes: List[ConstraintNode]) -> int:
        """计算树的最大深度"""
        if not root_nodes:
            return 0
        
        max_depth = 0
        for root in root_nodes:
            depth = self._get_node_depth(root)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _get_node_depth(self, node: ConstraintNode) -> int:
        """递归计算节点深度"""
        if not node.children:
            return node.level
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._get_node_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def get_nodes_by_level(self, level: int) -> List[ConstraintNode]:
        """
        获取指定层级的所有节点
        
        参数:
            level: 层级（0表示根节点）
            
        返回:
            List[ConstraintNode]: 该层级的所有节点
        """
        if not self.tree:
            return []
        
        nodes = []
        for node in self.tree.node_map.values():
            if node.level == level:
                nodes.append(node)
        
        return nodes
    
    def get_children_of_node(self, node_id: str) -> List[ConstraintNode]:
        """
        获取指定节点的所有子节点
        
        参数:
            node_id: 节点ID
            
        返回:
            List[ConstraintNode]: 子节点列表
        """
        if not self.tree or node_id not in self.tree.node_map:
            return []
        
        node = self.tree.node_map[node_id]
        return node.children
    
    def get_failed_nodes(self) -> List[ConstraintNode]:
        """
        获取所有失败的节点
        
        返回:
            List[ConstraintNode]: 失败节点列表
        """
        if not self.tree:
            return []
        
        failed_nodes = []
        for node in self.tree.node_map.values():
            if node.report and node.report.status == "fail":
                failed_nodes.append(node)
        
        return failed_nodes
    
    def update_node_report(self, node_id: str, report: AnalysisReport) -> None:
        """
        更新节点的检查报告
        
        参数:
            node_id: 节点ID
            report: 分析报告
        """
        if not self.tree or node_id not in self.tree.node_map:
            self.logger.warning(f"节点 {node_id} 不存在，无法更新报告")
            return
        
        node = self.tree.node_map[node_id]
        node.report = report
        report.node_id = node_id
        
        self.logger.debug(f"更新节点 {node_id} 的报告: {report.status}")
    
    def get_subtree_nodes(self, node_id: str) -> List[ConstraintNode]:
        """
        获取以指定节点为根的子树的所有节点（包括该节点）
        
        参数:
            node_id: 根节点ID
            
        返回:
            List[ConstraintNode]: 子树所有节点
        """
        if not self.tree or node_id not in self.tree.node_map:
            return []
        
        root_node = self.tree.node_map[node_id]
        result = [root_node]
        
        # 递归收集子节点
        def collect_children(node: ConstraintNode):
            for child in node.children:
                result.append(child)
                collect_children(child)
        
        collect_children(root_node)
        
        return result
    
    def get_path_to_root(self, node_id: str) -> List[ConstraintNode]:
        """
        获取从指定节点到根节点的路径
        
        参数:
            node_id: 节点ID
            
        返回:
            List[ConstraintNode]: 从根到该节点的路径
        """
        if not self.tree or node_id not in self.tree.node_map:
            return []
        
        path = []
        current_node = self.tree.node_map[node_id]
        
        while current_node:
            path.insert(0, current_node)
            if current_node.parent_id and current_node.parent_id in self.tree.node_map:
                current_node = self.tree.node_map[current_node.parent_id]
            else:
                break
        
        return path
    
    def get_unchecked_children_of_failed_nodes(self) -> List[ConstraintNode]:
        """
        获取所有失败节点的未检查子节点
        
        返回:
            List[ConstraintNode]: 未检查的子节点列表
        """
        if not self.tree:
            return []
        
        unchecked_children = []
        failed_nodes = self.get_failed_nodes()
        
        for failed_node in failed_nodes:
            for child in failed_node.children:
                # 如果子节点还没有检查报告，则加入列表
                if child.report is None:
                    unchecked_children.append(child)
        
        return unchecked_children
    
    def to_dict(self) -> Dict:
        """
        将树转换为字典格式（用于JSON序列化）
        
        返回:
            Dict: 树的字典表示
        """
        if not self.tree:
            return {}
        
        def node_to_dict(node: ConstraintNode) -> Dict:
            """递归将节点转换为字典"""
            node_dict = {
                "node_id": node.node_id,
                "name": node.constraint.name,
                "description": node.constraint.description,
                "level": node.level,
                "status": node.report.status if node.report else "unchecked",
                "children": [node_to_dict(child) for child in node.children]
            }
            
            if node.report:
                node_dict["summary"] = node.report.summary
                node_dict["violations_count"] = len(node.report.violations)
            
            return node_dict
        
        return {
            "max_depth": self.tree.max_depth,
            "total_nodes": self.tree.total_nodes,
            "root_count": len(self.tree.root_nodes),
            "roots": [node_to_dict(root) for root in self.tree.root_nodes]
        }
    
    def print_tree(self, include_status: bool = True) -> str:
        """
        生成树的文本表示（用于调试和日志）
        
        参数:
            include_status: 是否包含检查状态
            
        返回:
            str: 树的文本表示
        """
        if not self.tree:
            return "空树"
        
        lines = []
        lines.append(f"约束树 (深度: {self.tree.max_depth}, 节点数: {self.tree.total_nodes})")
        lines.append("=" * 80)
        
        def print_node(node: ConstraintNode, prefix: str = "", is_last: bool = True):
            """递归打印节点"""
            # 节点标记
            connector = "└── " if is_last else "├── "
            
            # 状态标记
            status_mark = ""
            if include_status and node.report:
                status_marks = {
                    "pass": "✅",
                    "fail": "❌",
                    "error": "⚠️",
                    "warning": "⚡"
                }
                status_mark = status_marks.get(node.report.status, "❓") + " "
            
            # 构建行
            line = f"{prefix}{connector}{status_mark}{node.constraint.name}"
            lines.append(line)
            
            # 递归打印子节点
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                print_node(child, prefix + extension, is_last_child)
        
        # 打印所有根节点
        for i, root in enumerate(self.tree.root_nodes):
            is_last_root = (i == len(self.tree.root_nodes) - 1)
            print_node(root, "", is_last_root)
        
        return "\n".join(lines)









from typing import List, Dict, Any, Optional
import time
import os
import logging
import json
from pathlib import Path

from .models import TrainingConfig, SchemaInfo, Constraint, AnalysisReport, RootCauseAnalysis
from .schema_analyzer import SchemaAnalyzer
from .database_executor import DatabaseExecutor
from .constraint_tree import ConstraintTreeManager
from .constraint_hierarchy_builder import ConstraintHierarchyBuilder
from .concurrent_executor import ConcurrentConstraintExecutor
from .root_cause_analyzer import RootCauseAnalyzer

# LLM Agent实现
from .generators import (
    LLMConstraintGenerator,
    LLMSQLGenerator,
    LLMResultAnalyzer,
    HybridConstraintGenerator,
    ConstraintGenerationMode
)
from .config import LLMConfig
from .llm_logger import create_new_logger_session, setup_unified_logging
from .llm_providers import LLMProviderFactory

# Token 追踪
from .token_tracker import get_token_tracker, get_consumption_manager, SQLTokenConsumptionManager

try:
    from .config_loader import get_model_config, list_available_specializations
    USE_AGENTS_CONFIG = True
except ImportError:
    USE_AGENTS_CONFIG = False


class LLMOrchestrator:
    """基于LLM Agent的核心流程编排器，使用LLM方法进行智能分析"""
    
    @classmethod
    def from_specialization(cls, 
                           specialization: str = "default",
                           constraint_generation_mode: ConstraintGenerationMode = ConstraintGenerationMode.PREDEFINED,
                           custom_constraints: Optional[Dict[str, List[Constraint]]] = None):
        """
        使用 agents 配置创建编排器（推荐方式）
        
        参数:
            specialization: 专门化类型（如 'default', 'test', 'conversation' 等）
            constraint_generation_mode: 约束生成模式
            custom_constraints: 自定义约束
            
        返回:
            LLMOrchestrator: 编排器实例
        """
        if not USE_AGENTS_CONFIG:
            raise RuntimeError("无法使用 agents 配置，请检查 agents/llm_config.yaml 文件")
        
        # 从 agents 配置加载
        model_config = get_model_config(specialization)
        
        # 转换为 LLMConfig
        llm_config = LLMConfig(
            model_name=model_config.model,
            api_key=model_config.api_key,
            timeout=model_config.timeout,
            temperature=0.1,
            max_tokens=4000
        )
        
        # 创建提供商
        provider = LLMProviderFactory.create_provider_from_specialization(specialization)
        
        # 创建编排器
        return cls(
            llm_provider=provider,
            llm_config=llm_config,
            constraint_generation_mode=constraint_generation_mode,
            custom_constraints=custom_constraints
        )
    
    def __init__(self, 
                 llm_provider,
                 llm_config: Optional[LLMConfig] = None,
                 constraint_generation_mode: ConstraintGenerationMode = ConstraintGenerationMode.PREDEFINED,
                 custom_constraints: Optional[Dict[str, List[Constraint]]] = None):
        """
        初始化编排器
        
        参数:
            llm_provider: LLM提供者实例
            llm_config: LLM配置
            constraint_generation_mode: 约束生成模式（predefined/llm/hybrid）
            custom_constraints: 自定义约束字典
        """
        self.llm_provider = llm_provider
        self.constraint_generation_mode = constraint_generation_mode
        self.custom_constraints = custom_constraints
        
        # 初始化LLM配置 - 如果没有提供配置，从提供者获取配置
        if llm_config is None:
            # 尝试从提供者获取配置，如果失败则使用DeepSeek V3默认配置
            if hasattr(llm_provider, 'config'):
                self.llm_config = llm_provider.config
            else:
                self.llm_config = LLMConfig(
                    model_name="deepseek-chat",
                    temperature=0.1,
                    max_tokens=2000,
                    base_url="https://api.deepseek.com"
                )
        else:
            self.llm_config = llm_config
        
        # 设置统一的日志系统（包括LLM日志和应用日志）
        log_file_path = setup_unified_logging()
        self.llm_logger = create_new_logger_session()
        
        # 初始化组件
        self._init_components()
        
        # 获取logger实例
        self.logger = logging.getLogger("SDCCheck.LLMOrchestrator")
        
        # Token 追踪和消耗管理
        self.token_tracker = get_token_tracker()
        self.consumption_manager = get_consumption_manager()
        
        self.logger.info(f"初始化编排器，约束生成模式: {constraint_generation_mode}，统一日志文件: {log_file_path}")
    
    def _init_components(self):
        """初始化各个组件"""
        # 数据库相关组件
        self.schema_analyzer = SchemaAnalyzer()
        self.db_executor = DatabaseExecutor()
        
        # 确定提供商名称
        provider_name = "openai"
        
        # 根据约束生成模式初始化约束生成器
        if self.constraint_generation_mode in [ConstraintGenerationMode.LLM, ConstraintGenerationMode.HYBRID]:
            # 需要LLM功能的模式
            self.constraint_generator = HybridConstraintGenerator(
                llm_config=self.llm_config,
                provider_name=provider_name,
                custom_constraints=self.custom_constraints,
                default_mode=self.constraint_generation_mode
            )
        else:
            # 纯预定义模式
            self.constraint_generator = HybridConstraintGenerator(
                llm_config=None,  # 不需要LLM配置
                provider_name=provider_name,
                custom_constraints=self.custom_constraints,
                default_mode=ConstraintGenerationMode.PREDEFINED
            )
        
        # SQL生成器和结果分析器仍然使用LLM（这些功能暂时保持LLM方式）
        self.sql_generator = LLMSQLGenerator(self.llm_config, provider_name)
        self.result_analyzer = LLMResultAnalyzer(self.llm_config, provider_name)
    
    def run_with_hierarchy(self, config: TrainingConfig, db_path: str, 
                          output_dir: Optional[str] = None,
                          max_workers: int = 10,
                          target_depth: int = 4,
                          script_path: Optional[str] = None,
                          use_dynamic_constraints: bool = True) -> RootCauseAnalysis:
        """
        使用层级树结构执行并发约束检查和根因分析
        
        参数:
            config: 训练配置
            db_path: 数据库路径
            output_dir: 输出目录（用于保存JSON报告）
            max_workers: 最大并发数（默认10）
            target_depth: 目标树深度（默认4）
            script_path: Megatron脚本路径（可选，用于自动解析配置）
            use_dynamic_constraints: 是否使用动态约束配置（默认True）
            
        返回:
            RootCauseAnalysis: 根因分析结果
        """
        start_time = time.time()
        self.logger.info(f"开始层级约束检查，数据库: {db_path}，配置: DP={config.dp}, TP={config.tp}")
        
        try:
            # 如果提供了脚本路径且启用动态约束，生成动态约束配置
            dynamic_constraints_path = None
            if script_path and use_dynamic_constraints:
                self.logger.info(f"步骤0: 从脚本生成动态约束配置...")
                try:
                    from .script_config_injector import ScriptConfigInjector
                    from pathlib import Path
                    
                    injector = ScriptConfigInjector()
                    
                    # 确定配置文件路径
                    current_dir = Path(__file__).parent
                    base_constraints = str(current_dir.parent / "config" / "predefined_constraints.json")
                    output_constraints = str(current_dir.parent / "config" / "dynamic_constraints.json")
                    
                    # 生成动态约束
                    script_config = injector.inject_script_config_to_constraints(
                        script_path,
                        base_constraints,
                        output_constraints
                    )
                    
                    dynamic_constraints_path = output_constraints
                    self.logger.info(f"动态约束配置已生成: {dynamic_constraints_path}")
                    self.logger.info(f"脚本配置: TP={script_config.tp}, PP={script_config.pp}, DP={script_config.dp}")
                    
                    # 更新 config 对象（使用脚本解析的配置）
                    config = TrainingConfig(
                        dp=script_config.dp,
                        tp=script_config.tp,
                        pp=script_config.pp,
                        ep=config.ep,
                        sp=config.sp,
                        zero_stage=1 if script_config.use_distributed_optimizer else 0
                    )
                    
                except Exception as e:
                    self.logger.warning(f"生成动态约束失败，将使用默认约束: {e}")
            
            # 步骤1: 分析数据库模式
            self.logger.info("步骤1: 分析数据库模式...")
            schema_info = self.schema_analyzer.analyze(db_path)
            
            # 步骤2: 生成约束（使用动态约束配置或默认配置）
            self.logger.info(f"步骤2: 生成约束...")
            
            # 如果有动态约束配置，使用它重新初始化约束生成器
            if dynamic_constraints_path and Path(dynamic_constraints_path).exists():
                from .generators import PredefinedConstraintGenerator
                self.logger.info(f"使用动态约束配置: {dynamic_constraints_path}")
                temp_generator = PredefinedConstraintGenerator(config_path=dynamic_constraints_path)
                constraints = temp_generator.generate(schema_info, config)
            else:
                self.logger.info("使用默认约束配置")
                constraints = self.constraint_generator.generate(schema_info, config)
            
            if not constraints:
                self.logger.warning("未找到适用的约束")
                return self._create_empty_analysis()
            
            self.logger.info(f"生成了{len(constraints)}个约束")
            
            # 步骤3: 使用AI构建约束层级结构
            self.logger.info(f"步骤3: 构建约束层级结构（目标深度{target_depth}级）...")
            hierarchy_builder = ConstraintHierarchyBuilder(self.llm_config, "openai")
            hierarchical_constraints = hierarchy_builder.build_hierarchy(constraints, target_depth)
            
            if not hierarchy_builder.validate_hierarchy(hierarchical_constraints):
                self.logger.warning("层级结构验证失败，回退到扁平结构")
                hierarchical_constraints = constraints
            
            # 步骤4: 构建约束树
            self.logger.info("步骤4: 构建约束树...")
            tree_manager = ConstraintTreeManager()
            tree = tree_manager.build_tree_from_constraints(hierarchical_constraints)
            
            self.logger.info(f"约束树构建完成:\n{tree_manager.print_tree(include_status=False)}")
            
            # 步骤5: 连接数据库
            self.db_executor.connect(db_path)
            
            # 步骤6: 创建并发执行器
            self.logger.info(f"步骤5: 创建并发执行器（并发数={max_workers}）...")
            concurrent_executor = ConcurrentConstraintExecutor(
                db_executor=self.db_executor,
                sql_generator=self.sql_generator,
                result_analyzer=self.result_analyzer,
                max_workers=max_workers
            )
            
            # 步骤7: 逐层执行约束树（失败节点的子节点会被继续执行）
            self.logger.info("步骤6: 开始逐层并发执行约束...")
            execution_stats = concurrent_executor.execute_tree_level_by_level(
                tree_manager=tree_manager,
                start_level=0,
                stop_on_all_pass=False  # 不在全部通过时停止
            )
            
            self.logger.info(f"约束树执行完成:\n{tree_manager.print_tree(include_status=True)}")
            
            # 步骤8: 根因分析
            self.logger.info("步骤7: 进行根因分析...")
            root_cause_analyzer = RootCauseAnalyzer()
            analysis = root_cause_analyzer.analyze(tree_manager)
            
            # 打印分析摘要
            summary_text = root_cause_analyzer.print_summary(analysis)
            print(summary_text)
            self.logger.info(f"根因分析完成:\n{summary_text}")
            
            # 关闭数据库连接
            self.db_executor.close()
            
            # 保存JSON报告
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                json_file = output_path / f"root_cause_analysis_{analysis.analysis_id}.json"
                root_cause_analyzer.export_to_json(analysis, str(json_file))
                self.logger.info(f"根因分析报告已保存: {json_file}")
            
            end_time = time.time()
            total_time = end_time - start_time
            self.logger.info(f"层级约束检查完成，总耗时: {total_time:.2f}秒")
            
            # Token 消耗汇总 - 从全局 TokenTracker 获取所有并发执行的总消耗
            token_usage = self.token_tracker.get_usage()
            token_summary = f"""
╔════════════════════════════════════════════════════════════════╗
║              SDCCheck 层级检查 Token 消耗汇总                   ║
╠════════════════════════════════════════════════════════════════╣
║  总 Prompt Tokens:    {token_usage['prompt_tokens']:>10}                          ║
║  总 Completion Tokens:{token_usage['completion_tokens']:>10}                          ║
║  总 Token 消耗:       {token_usage['total_tokens']:>10}                          ║
║  总 LLM 调用次数:     {token_usage['call_count']:>10}                          ║
║  执行耗时(秒):        {total_time:>10.2f}                          ║
╚════════════════════════════════════════════════════════════════╝
"""
            print(token_summary)
            self.logger.info(f"Token 消耗汇总: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}, calls={token_usage['call_count']}")
            
            # 保存 Token 消耗汇总到文件
            if output_dir:
                token_report = {
                    "analysis_id": analysis.analysis_id,
                    "timestamp": token_usage.get('end_time', ''),
                    "execution_time_seconds": round(total_time, 2),
                    "token_consumption": {
                        "prompt_tokens": token_usage['prompt_tokens'],
                        "completion_tokens": token_usage['completion_tokens'],
                        "total_tokens": token_usage['total_tokens'],
                        "llm_call_count": token_usage['call_count'],
                        "model": token_usage.get('model', ''),
                    },
                    "constraint_statistics": {
                        "total_constraints": analysis.total_constraints,
                        "passed_constraints": analysis.passed_constraints,
                        "failed_constraints": analysis.failed_constraints,
                    }
                }
                token_file = output_path / f"token_consumption_{analysis.analysis_id}.json"
                with open(token_file, 'w', encoding='utf-8') as f:
                    json.dump(token_report, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Token 消耗报告已保存: {token_file}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"层级检查过程中出错: {e}", exc_info=True)
            
            # 确保数据库连接已关闭
            try:
                self.db_executor.close()
            except:
                pass
            
            raise
    
    def _create_empty_analysis(self) -> RootCauseAnalysis:
        """创建空的根因分析结果"""
        return RootCauseAnalysis(
            analysis_id=f"EMPTY_{int(time.time())}",
            total_constraints=0,
            passed_constraints=0,
            failed_constraints=0,
            failure_chains=[],
            root_causes=[],
            recommendations=["未找到适用的约束"],
            constraint_tree={},
            execution_summary={}
        )
    
    def run(self, config: TrainingConfig, db_path: str) -> List[AnalysisReport]:
        """
        执行完整的检查流程
        
        参数:
            config: 训练配置
            db_path: 数据库路径
            
        返回:
            List[AnalysisReport]: 所有约束的分析报告
        """
        start_time = time.time()
        self.logger.info(f"开始检查，数据库: {db_path}，配置: DP={config.dp}, TP={config.tp}")

        # Incremental dump support: if SDC_INCREMENTAL_REPORTS is set,
        # rewrite that JSON after every constraint so a killed run still
        # produces partial results.
        incremental_path = os.environ.get("SDC_INCREMENTAL_REPORTS")

        def _norm_status(s):
            v = str(s).split(".")[-1].lower()
            if v in ("pass", "fail", "error", "warning"):
                return v
            return v or "unknown"

        def _norm_type(t):
            return str(t).split(".")[-1].lower()

        def _dump_incremental(reports_list):
            if not incremental_path:
                return
            try:
                out = []
                for r in reports_list:
                    out.append({
                        "constraint_name": getattr(r.constraint, "name", str(r.constraint)),
                        "constraint_type": _norm_type(getattr(r.constraint, "type", "")),
                        "status": _norm_status(r.status),
                        "summary": r.summary,
                        "n_violations": len(r.violations) if r.violations else 0,
                    })
                tmp = incremental_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(out, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, incremental_path)
            except Exception as _e:
                self.logger.warning(f"incremental dump failed: {_e}")

        try:
            # 步骤1: 分析数据库模式
            self.logger.info("步骤1: 分析数据库模式...")
            schema_info = self.schema_analyzer.analyze(db_path)
            
            # 步骤2: 生成约束
            mode_desc = {
                ConstraintGenerationMode.PREDEFINED: "预定义约束",
                ConstraintGenerationMode.LLM: "LLM生成",
                ConstraintGenerationMode.HYBRID: "混合模式"
            }.get(self.constraint_generation_mode, "未知模式")
            
            self.logger.info(f"步骤2: 生成约束（使用{mode_desc}）...")
            constraints = self.constraint_generator.generate(schema_info, config)
            
            if not constraints:
                self.logger.warning("未找到适用的约束，请检查数据库结构或训练配置")
                return []
                
            self.logger.info(f"生成了{len(constraints)}个约束待检查")
            
            # 步骤3: 连接数据库
            self.db_executor.connect(db_path)
            
            # 步骤4: 对每个约束执行检查
            reports = []
            for i, constraint in enumerate(constraints, 1):
                self.logger.info(f"检查约束 {i}/{len(constraints)}: {constraint.name}")
                
                # Token 追踪：重置统计，开始新一轮追踪
                self.token_tracker.reset()
                self.token_tracker.set_context(constraint_name=constraint.name)
                
                try:
                    # 步骤4a: 生成SQL
                    self.logger.debug("生成SQL（使用LLM方法）")
                    sql = self.sql_generator.generate(constraint)
                    self.logger.debug(f"生成的SQL: {sql}")
                    
                    # 步骤4b: 执行SQL
                    result = self.db_executor.execute(sql)
                    
                    # 步骤4c: 分析结果
                    self.logger.debug("分析结果（使用LLM方法）")
                    report = self.result_analyzer.analyze(constraint, result)
                    reports.append(report)
                    
                    # 记录结果
                    self.logger.info(f"约束[{constraint.name}] 检查结果: {report.status} - {report.summary}")
                    _dump_incremental(reports)
                    
                    # Token 追踪：记录本次约束验证的 token 消耗
                    token_usage = self.token_tracker.get_usage()
                    self.consumption_manager.record_consumption(
                        constraint_name=constraint.name,
                        token_usage=token_usage,
                        sql=sql,
                        analysis_summary=report.summary
                    )
                    self.logger.info(f"约束[{constraint.name}] Token消耗: {token_usage.get('total_tokens', 0)}")
                    
                except Exception as e:
                    self.logger.error(f"检查约束[{constraint.name}]时出错: {e}")
                    # 创建错误报告
                    error_report = AnalysisReport(
                        constraint=constraint,
                        status="error",
                        summary=f"检查过程中出错: {str(e)}",
                        violations=[],
                        raw_data=None
                    )
                    reports.append(error_report)
                    _dump_incremental(reports)

                    # Token 追踪：即使出错也记录消耗
                    token_usage = self.token_tracker.get_usage()
                    if token_usage.get('total_tokens', 0) > 0:
                        self.consumption_manager.record_consumption(
                            constraint_name=constraint.name,
                            token_usage=token_usage,
                            sql="",
                            analysis_summary=f"错误: {str(e)}"
                        )
                    continue
            
            # 关闭数据库连接
            self.db_executor.close()
            
            end_time = time.time()
            self.logger.info(f"检查完成，耗时: {end_time - start_time:.2f}秒")
            
            # 打印 Token 消耗汇总
            token_summary = self.consumption_manager.print_summary()
            print(token_summary)
            self.logger.info(f"Token 消耗汇总:\n{token_summary}")
            
            return reports
            
        except Exception as e:
            self.logger.error(f"检查过程中出错: {e}", exc_info=True)
            
            # 确保数据库连接已关闭
            try:
                self.db_executor.close()
            except:
                pass
                
            raise
    
    def update_llm_provider(self, llm_provider, llm_config: Optional[LLMConfig] = None) -> None:
        """
        更新LLM提供者并重新初始化相关组件
        
        参数:
            llm_provider: 新的LLM提供者实例
            llm_config: 新的LLM配置（可选）
        """
        self.llm_provider = llm_provider
        if llm_config:
            self.llm_config = llm_config
        
        # 确定提供商名称
        provider_name = "deepseek" if "deepseek" in str(type(self.llm_provider).__name__).lower() else "mock"
        
        # 重新初始化LLM组件
        self.constraint_generator = LLMConstraintGenerator(self.llm_config, provider_name)
        self.sql_generator = LLMSQLGenerator(self.llm_config, provider_name)
        self.result_analyzer = LLMResultAnalyzer(self.llm_config, provider_name)
        
        self.logger.info(f"已更新LLM提供者({provider_name})并重新初始化组件")
    
    def set_constraint_generation_mode(self, mode: ConstraintGenerationMode) -> None:
        """
        设置约束生成模式并重新初始化约束生成器
        
        参数:
            mode: 新的约束生成模式
        """
        self.constraint_generation_mode = mode
        
        # 重新初始化约束生成器
        provider_name = "deepseek" if "deepseek" in str(type(self.llm_provider).__name__).lower() else "mock"
        
        if mode in [ConstraintGenerationMode.LLM, ConstraintGenerationMode.HYBRID]:
            self.constraint_generator = HybridConstraintGenerator(
                llm_config=self.llm_config,
                provider_name=provider_name,
                custom_constraints=self.custom_constraints,
                default_mode=mode
            )
        else:
            self.constraint_generator = HybridConstraintGenerator(
                llm_config=None,
                provider_name=provider_name,
                custom_constraints=self.custom_constraints,
                default_mode=ConstraintGenerationMode.PREDEFINED
            )
        
        self.logger.info(f"约束生成模式已设置为: {mode}")
    
    def get_constraint_generation_mode(self) -> ConstraintGenerationMode:
        """
        获取当前的约束生成模式
        
        返回:
            ConstraintGenerationMode: 当前模式
        """
        return self.constraint_generation_mode
    
    def list_available_predefined_constraints(self) -> Dict[str, List[str]]:
        """
        列出所有可用的预定义约束
        
        返回:
            Dict[str, List[str]]: 按类别组织的约束名称列表
        """
        if hasattr(self.constraint_generator, 'list_available_predefined_constraints'):
            return self.constraint_generator.list_available_predefined_constraints()
        else:
            return {}
    
    def add_custom_constraint(self, category: str, constraint: Constraint) -> None:
        """
        添加自定义约束
        
        参数:
            category: 约束类别
            constraint: 约束对象
        """
        if hasattr(self.constraint_generator, 'add_custom_constraint'):
            self.constraint_generator.add_custom_constraint(category, constraint)
            self.logger.info(f"添加自定义约束 '{constraint.name}' 到类别 '{category}'")
        else:
            self.logger.warning("当前约束生成器不支持添加自定义约束")
    
    def print_reports(self, reports: List[AnalysisReport]) -> None:
        """
        打印分析报告
        
        参数:
            reports: 分析报告列表
        """
        if not reports:
            print("没有生成报告")
            return
            
        print("\n" + "=" * 80)
        print(f"{'约束名称':<20}{'状态':<10}{'摘要'}")
        print("-" * 80)
        
        for report in reports:
            status_str = {
                "pass": "✅ 通过",
                "fail": "❌ 失败",
                "error": "⚠️ 错误",
                "warning": "⚠️ 警告"
            }.get(report.status, report.status)
            
            print(f"{report.constraint.name:<20}{status_str:<10}{report.summary}")
            
            if report.violations:
                print("\n违规示例:")
                for i, violation in enumerate(report.violations[:3]):  # 最多显示3条
                    print(f"  {i+1}. {violation}")
                    
            if report.suggestion:
                print(f"\n建议: {report.suggestion}")
                
            print("-" * 80)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        返回:
            Dict[str, Any]: 性能统计信息
        """
        constraint_stats = {}
        if hasattr(self.constraint_generator, 'get_generation_stats'):
            constraint_stats = self.constraint_generator.get_generation_stats()
        
        return {
            "methods": {
                "constraint_generation": self.constraint_generation_mode.value,
                "sql_generation": "LLM",
                "result_analysis": "LLM"
            },
            "llm_provider": str(type(self.llm_provider).__name__),
            "constraint_generation_mode": self.constraint_generation_mode.value,
            "constraint_generator_stats": constraint_stats
        }
    
    def export_analysis_to_json(self, analysis: RootCauseAnalysis, output_path: str) -> None:
        """
        导出根因分析结果到JSON文件
        
        参数:
            analysis: 根因分析结果
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"根因分析结果已导出到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"导出JSON失败: {e}")
            raise
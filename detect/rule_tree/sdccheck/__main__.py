#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from .models import TrainingConfig
from .llm_orchestrator import LLMOrchestrator
from .llm_providers import LLMProviderFactory
from .config import LLMConfig


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="SDCCheck - 分布式训练检查点一致性验证工具"
    )
    
    parser.add_argument(
        "db_path",
        type=str,
        help="数据库文件路径"
    )
    
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="数据并行度 (默认: 1)"
    )
    
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="张量并行度 (默认: 1)"
    )
    
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="流水线并行度 (默认: 1)"
    )
    
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="专家并行度 (默认: 1)"
    )
    
    parser.add_argument(
        "--sp",
        type=int,
        default=1,
        help="序列并行度 (默认: 1)"
    )
    
    parser.add_argument(
        "--zero-stage",
        type=int,
        default=0,
        dest="zero_stage",
        help="ZeRO优化阶段 (默认: 0)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细日志信息"
    )

    parser.add_argument(
        "--constraints-file",
        type=str,
        default=None,
        help="自定义约束库 JSON 文件路径 (用于消融实验)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="LLM 提供商: mock / openai / deepseek (默认 mock)"
    )

    parser.add_argument(
        "--reports-out",
        type=str,
        default=None,
        help="将每条约束的检查结果落盘到该 JSON 文件 (用于消融汇总)"
    )

    args = parser.parse_args()

    import os
    if args.constraints_file:
        os.environ["SDC_CONSTRAINTS_FILE"] = args.constraints_file
        print(f"使用自定义约束库: {args.constraints_file}")
    if args.reports_out:
        # Mirror reports-out to the orchestrator's incremental-dump hook so
        # mid-run kills still produce usable partial reports.
        os.environ["SDC_INCREMENTAL_REPORTS"] = args.reports_out
    
    # 检查数据库文件是否存在
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"错误: 数据库文件不存在: {db_path}")
        sys.exit(1)
        
    # 创建训练配置
    config = TrainingConfig(
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        sp=args.sp,
        zero_stage=args.zero_stage
    )
    
    print(f"开始检查...")
    print(f"数据库: {db_path}")
    print(f"训练配置: DP={config.dp}, TP={config.tp}, PP={config.pp}, EP={config.ep}, "
          f"SP={config.sp}, ZeRO={config.zero_stage}")
    
    # 运行检查
    if args.provider == "mock":
        llm_config = LLMConfig(model_name="mock-model", temperature=0.1, max_tokens=2000)
        llm_provider = LLMProviderFactory.create_provider("mock", llm_config)
    else:
        # 从 specialization 配置加载真实 provider (DeepSeek 等)
        try:
            llm_provider = LLMProviderFactory.create_provider_from_specialization("default")
            llm_config = llm_provider.config if hasattr(llm_provider, "config") else \
                LLMConfig(model_name="deepseek-chat", temperature=0.1, max_tokens=2000)
        except Exception as e:
            print(f"加载 specialization provider 失败 ({e})，回退到 mock")
            llm_config = LLMConfig(model_name="mock-model", temperature=0.1, max_tokens=2000)
            llm_provider = LLMProviderFactory.create_provider("mock", llm_config)
    orchestrator = LLMOrchestrator(llm_provider)
    try:
        reports = orchestrator.run(config, str(db_path))
        orchestrator.print_reports(reports)
    except Exception as e:
        print(f"检查过程中出错: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    if args.reports_out and reports:
        import json as _json
        out = []
        for r in reports:
            out.append({
                "constraint_name": getattr(r.constraint, "name", str(r.constraint)),
                "constraint_type": str(getattr(r.constraint, "type", "")),
                "status": r.status,
                "summary": r.summary,
                "n_violations": len(r.violations) if r.violations else 0,
            })
        with open(args.reports_out, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"reports written: {args.reports_out}")
        
    # 根据检查结果设置退出码
    if not reports:
        print("未执行任何检查")
        sys.exit(2)
        
    # 如果有任何失败或错误的约束，返回非零退出码
    if any(report.status in ("fail", "error") for report in reports):
        sys.exit(3)
        

if __name__ == "__main__":
    main()
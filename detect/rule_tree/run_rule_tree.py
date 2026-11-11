"""Run the compiled constraint tree. Coarse nodes first; children of a failure run next."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sdccheck.config import LLMConfig
from sdccheck.llm_orchestrator import LLMOrchestrator
from sdccheck.llm_providers import LLMProviderFactory
from sdccheck.models import TrainingConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path")
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--sp", type=int, default=1)
    ap.add_argument("--zero-stage", type=int, default=0, dest="zero_stage")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not os.path.exists(a.db_path):
        raise SystemExit(f"missing db {a.db_path}")
    config = TrainingConfig(
        dp=a.dp, tp=a.tp, pp=a.pp, ep=a.ep, sp=a.sp, zero_stage=a.zero_stage,
    )
    if a.provider == "mock":
        llm_config = LLMConfig(model_name="mock-model", temperature=0.1, max_tokens=2000)
        provider = LLMProviderFactory.create_provider("mock", llm_config)
    else:
        provider = LLMProviderFactory.create_provider_from_specialization("default")
        llm_config = getattr(provider, "config", None)
    print(
        f"rule tree db={a.db_path} dp={config.dp} tp={config.tp} pp={config.pp} "
        f"workers={a.workers} depth={a.depth} provider={a.provider}",
        flush=True,
    )
    orch = LLMOrchestrator(provider, llm_config)
    analysis = orch.run_with_hierarchy(
        config, a.db_path, output_dir=a.out, max_workers=a.workers, target_depth=a.depth,
    )
    print(
        f"rule tree done failed={analysis.failed_constraints} "
        f"passed={analysis.passed_constraints} total={analysis.total_constraints}",
        flush=True,
    )
    return 0 if analysis.failed_constraints == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

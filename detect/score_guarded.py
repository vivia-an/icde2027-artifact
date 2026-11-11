"""Apply the frozen guarded specs to one GPU job directory.

The job is produced by detect_gpu.py. This step does not train. It reads rank*.duckdb
and runs the offline specs in catalog/check_specs/specs/<layout>/guarded.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "py"))

from trainaudit.gcq import views_layout_v3 as V  # noqa: E402
from trainaudit.gcq.evaluate import run_compiled  # noqa: E402
from trainaudit.gcq.forms_v3 import compile_rule_v3, load_any_v3  # noqa: E402


def specs_for(layout):
    spec_dir = os.path.join(ROOT, "catalog", "check_specs", "specs", layout, "guarded")
    out = []
    for path in sorted(glob.glob(os.path.join(spec_dir, "*.json"))):
        spec = load_any_v3(path)
        out.append((spec, compile_rule_v3(spec)))
    if not out:
        raise SystemExit(f"no guarded specs for {layout}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--run", required=True, help="directory written by detect_gpu.py")
    a = ap.parse_args()
    gt_path = os.path.join(a.run, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)
    dbs = sorted(glob.glob(os.path.join(a.run, "rank*.duckdb")))
    if not dbs:
        raise SystemExit(f"no rank traces in {a.run}")
    con = V.open_job(dbs)
    inj_step = gt.get("step")
    total = 0
    for spec, compiled in specs_for(a.layout):
        res = run_compiled(con, compiled)
        n = 0
        earliest = None
        if res["status"] == "ok":
            for w in res["witnesses"]:
                step = int(w[0])
                if inj_step is not None and step < inj_step:
                    continue
                n += 1
                earliest = step if earliest is None else min(earliest, step)
        total += n
        print(f"{spec.id}\t{res['status']}\tviolations={n}\tearliest={earliest}")
    con.close()
    print(f"detected={total > 0}\ttotal={total}")
    return 0 if True else 1


if __name__ == "__main__":
    sys.exit(main())

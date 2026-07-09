# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Cross-seed ON-vs-OFF analysis (EVAL_FRAMEWORK §0a, §3): aggregate the
multi-seed experiment (run_experiment.sh) and apply the pre-registered decision
rule to manager-vs-scripted and manager-vs-control on the PRIMARY metric
(held-out AUC = sample efficiency), against the MEASURED noise band tau.

Usage: analyze_experiment.py <experiment_dir> [--tau-json <chaos>/tau.json]
  <experiment_dir> holds control_seed*.json, manager_seed*.json, scripted_seed*.json."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional


def _auc(journal) -> Optional[float]:
    s = [(e.get("heldout") or {}).get("heldout_success_rate")
         for e in journal if "segment" in e]
    s = [x for x in s if x is not None]
    return sum(s) / len(s) if s else None


def _seeds(exp_dir: str, arm: str) -> Dict[str, float]:
    out = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, f"{arm}_seed*.json"))):
        m = re.search(r"_seed(\w+)\.json$", f)
        if not m:
            continue
        auc = _auc(json.load(open(f)))
        if auc is not None:
            out[m.group(1)] = auc
    return out


def _load_tau(path: Optional[str]) -> Optional[float]:
    if path and os.path.exists(path):
        return json.load(open(path)).get("tau")
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-seed ON-vs-OFF analysis")
    p.add_argument("experiment_dir")
    p.add_argument("--tau-json", default=None,
                   help="tau.json from compute_tau.py; else reports deltas only")
    args = p.parse_args(argv)

    mgr = _seeds(args.experiment_dir, "manager")
    scr = _seeds(args.experiment_dir, "scripted")
    ctl = _seeds(args.experiment_dir, "control")
    if not mgr:
        print(f"no manager_seed*.json in {args.experiment_dir}")
        return 2

    print("=" * 68)
    print("CROSS-SEED ON-vs-OFF  (primary metric: held-out AUC = sample efficiency)")
    print("=" * 68)
    seeds = sorted(set(mgr) | set(scr) | set(ctl))
    print(f"\n{'seed':>8} {'control':>9} {'manager':>9} {'scripted':>9} "
          f"{'mgr-scr':>9} {'mgr-ctl':>9}")
    m_s, m_c = [], []
    for s in seeds:
        c, m, sc = ctl.get(s), mgr.get(s), scr.get(s)
        dms = (m - sc) if (m is not None and sc is not None) else None
        dmc = (m - c) if (m is not None and c is not None) else None
        if dms is not None:
            m_s.append(dms)
        if dmc is not None:
            m_c.append(dmc)
        fmt = lambda x: f"{x:.4f}" if x is not None else "  n/a  "
        fmtd = lambda x: f"{x:+.4f}" if x is not None else "  n/a  "
        print(f"{s:>8} {fmt(c):>9} {fmt(m):>9} {fmt(sc):>9} {fmtd(dms):>9} {fmtd(dmc):>9}")

    tau = _load_tau(args.tau_json)
    print("\n--- verdict (pre-registered decision rule, EVAL_FRAMEWORK §3) ---")
    if not m_s:
        print("  no manager/scripted seed pairs to compare")
        return 1
    mean_ms = statistics.mean(m_s)
    sd_ms = statistics.pstdev(m_s) if len(m_s) > 1 else 0.0
    same_sign = all(x > 0 for x in m_s) or all(x < 0 for x in m_s)
    print(f"  manager - scripted (AUC): mean {mean_ms:+.4f}  sd {sd_ms:.4f}  "
          f"n={len(m_s)}  same-sign-across-seeds={same_sign}")
    if m_c:
        print(f"  manager - control  (AUC): mean {statistics.mean(m_c):+.4f}  n={len(m_c)}")

    if tau is None:
        print("\n  tau NOT provided (--tau-json). Report deltas only; NO verdict without a")
        print("  measured Isaac Lab noise band. Run run_chaos_probe.sh + compute_tau.py.")
        return 0

    print(f"\n  tau (measured) = {tau:.4f}")
    if abs(mean_ms) <= tau:
        print("  VERDICT: manager ≈ scripted within tau -> NO measured benefit of adaptivity")
        print("           (the SONIC-consistent null; a valid, honest finding).")
    elif mean_ms > tau and same_sign:
        print("  VERDICT: manager > scripted by > tau, consistent across seeds -> adaptivity HELPS")
    elif mean_ms < -tau and same_sign:
        print("  VERDICT: manager < scripted by > tau -> adaptivity HURTS (real finding)")
    else:
        print("  VERDICT: |manager - scripted| > tau but sign INCONSISTENT across seeds")
        print("           -> underpowered / noisy; add seeds before claiming anything.")
    print("\n  NOTE: honest n. Cross-seed sign-consistency + AUC (not final) + a MEASURED tau")
    print("  are all required; a single seed cannot support any row (same-seed mgr==scr by")
    print("  construction — see EVAL_FRAMEWORK §0a).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

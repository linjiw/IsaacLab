# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Compute the Isaac Lab noise band + equivalence tolerance tau from a chaos
probe (EVAL_FRAMEWORK §2). Reads the per-seed control journals produced by
run_chaos_probe.sh and reports:
  - final heldout_success_rate per seed
  - sigma_noise (std across seeds) and the relative floor sigma/mean
  - tau = calibrate_tau(relative_floor, safety_factor=3)
This tau is what analyze_pilot.py / EquivalenceGate should use for a REAL
Isaac Lab verdict (replacing the SONIC placeholder).

Usage: compute_tau.py <chaos_dir>   (holds journal_seed*.json)"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def _final_heldout(journal):
    vals = [(e.get("heldout") or {}).get("heldout_success_rate")
            for e in journal if "segment" in e]
    vals = [v for v in vals if v is not None]
    return vals[-1] if vals else None


def _last2_mean(journal):
    vals = [(e.get("heldout") or {}).get("heldout_success_rate")
            for e in journal if "segment" in e]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals[-2:]) if len(vals) >= 2 else (vals[-1] if vals else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute Isaac Lab tau from a chaos probe")
    p.add_argument("chaos_dir")
    p.add_argument("--safety-factor", type=float, default=3.0)
    p.add_argument("--min-effect", type=float, default=None,
                   help="smallest held-out effect we want to detect; tau must "
                        "stay below it (calibrate_tau raises otherwise)")
    args = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.chaos_dir, "journal_seed*.json")))
    if len(files) < 2:
        print(f"need >=2 seed journals in {args.chaos_dir}, found {len(files)}")
        return 2

    finals, last2 = [], []
    for f in files:
        j = json.load(open(f))
        fin = _final_heldout(j)
        l2 = _last2_mean(j)
        seed = os.path.basename(f).replace("journal_seed", "").replace(".json", "")
        print(f"  seed {seed}: final_heldout={fin:.4f}  last2_mean={l2:.4f}")
        if fin is not None:
            finals.append(fin)
        if l2 is not None:
            last2.append(l2)

    mean_f = statistics.mean(finals)
    sigma_f = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    rel_floor = (sigma_f / mean_f) if mean_f > 0 else 0.0
    print(f"\n  n={len(finals)}  mean_final={mean_f:.4f}  "
          f"sigma_noise={sigma_f:.4f}  relative_floor={rel_floor:.4f}")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "runmanager"))
    from core.equivalence import calibrate_tau
    try:
        tau = calibrate_tau(rel_floor if rel_floor > 0 else 1e-6,
                            min_effect_dev=args.min_effect,
                            safety_factor=args.safety_factor)
        print(f"\n  >>> Isaac Lab tau = {tau:.4f} "
              f"(safety_factor={args.safety_factor}) <<<")
        print(f"  Use: EquivalenceGate(tau={tau:.4f}) for manager-vs-scripted "
              "verdicts on this engine/config.")
        # persist for analyze_pilot / the real experiment
        out = os.path.join(args.chaos_dir, "tau.json")
        json.dump({"tau": tau, "sigma_noise": sigma_f, "mean_final": mean_f,
                   "relative_floor": rel_floor, "n_seeds": len(finals),
                   "safety_factor": args.safety_factor,
                   "finals": finals}, open(out, "w"), indent=2)
        print(f"  wrote {out}")
    except ValueError as e:
        print(f"\n  calibrate_tau refused: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

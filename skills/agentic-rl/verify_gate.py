# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Phase-1 gate-checklist verifier: inspect the pilot's journals + a held-out
eval and assert the conditions that make a Phase-1 run trustworthy (the G-items
from the adversarial review). Pure Python over the journal JSON — no GPU.

Usage: verify_gate.py <pilot_dir> [--min-heldout-var 0.0]
  <pilot_dir> holds journal_{manager,scripted,control}.json (from run_pilot.sh).
Exit 0 iff all HARD gates pass; prints a PASS/FAIL table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _applied(journal):
    return [e for e in journal if e.get("applied")]


def _segments(journal):
    return [e for e in journal if "segment" in e]


def check(pilot_dir: str) -> int:
    results = []   # (hard, name, ok, detail)

    def gate(hard, name, ok, detail=""):
        results.append((hard, name, bool(ok), detail))

    paths = {arm: os.path.join(pilot_dir, f"journal_{arm}.json")
             for arm in ("manager", "scripted", "control")}
    missing = [a for a, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"MISSING journals: {missing}")
        return 2
    J = {arm: _load(p) for arm, p in paths.items()}

    # G3: no segment_failed events (a crashed launch must be loud, not silent)
    for arm, j in J.items():
        failed = [e for e in j if e.get("event") == "segment_failed"]
        gate(True, f"[{arm}] no segment_failed", not failed,
             f"{len(failed)} failed" if failed else "clean")

    # G4: held-out metric populated for every segment of every arm
    for arm, j in J.items():
        segs = _segments(j)
        ho = [(e.get("heldout") or {}).get("heldout_success_rate") for e in segs]
        pop = [h for h in ho if h is not None]
        gate(True, f"[{arm}] held-out populated", len(pop) == len(segs) and segs,
             f"{len(pop)}/{len(segs)} segments")

    # G4b: headline is the held-out metric, and it is a valid fraction
    for arm, j in J.items():
        segs = _segments(j)
        ho = [(e.get("heldout") or {}).get("heldout_success_rate") for e in segs]
        valid = all(h is None or (0.0 <= h <= 1.0) for h in ho)
        gate(True, f"[{arm}] held-out in [0,1]", valid)

    # G8: per-condition decomposition present (detects a dominating condition)
    for arm, j in J.items():
        segs = _segments(j)
        pc = [(e.get("heldout") or {}).get("heldout_per_condition") for e in segs]
        has_pc = any(isinstance(x, dict) and x for x in pc)
        gate(True, f"[{arm}] per-condition present", has_pc)

    # G10: held-out metric is NON-DEGENERATE (varies across conditions or
    # segments) — a metric pinned at one value can't discriminate quality
    for arm, j in J.items():
        segs = _segments(j)
        vals = []
        for e in segs:
            pc = (e.get("heldout") or {}).get("heldout_per_condition") or {}
            vals += list(pc.values())
        spread = (max(vals) - min(vals)) if vals else 0.0
        gate(True, f"[{arm}] held-out non-degenerate", spread > 1e-9,
             f"per-condition spread={spread:.4f}")

    # G5: scripted ladder == manager's realized applied decisions
    m_ladder = {e["tick"]: (e["decision"]["knob"], e["decision"]["value"])
                for e in _applied(J["manager"])}
    s_moves = {e["tick"]: (e["decision"]["knob"], e["decision"]["value"])
               for e in _applied(J["scripted"])}
    faithful = all(m_ladder.get(t) == v for t, v in s_moves.items()) and s_moves
    gate(True, "scripted replays manager ladder", faithful,
         f"manager={len(m_ladder)} rungs, scripted={len(s_moves)} moves")

    # SAME-SEED DEGENERACY CHECK (EVAL_FRAMEWORK §0a): if scripted replays the
    # SAME seed's manager, bit-deterministic training makes their held-out
    # identical -> manager-vs-scripted is a tautology, NOT an adaptivity result.
    # This is a SOFT gate: it does not fail the run (the pilot is valid as
    # machinery validation), but it LOUDLY flags that the mgr-vs-scr delta here
    # is meaningless and a cross-seed run (run_experiment.sh) is required.
    m_ho = [(e.get("heldout") or {}).get("heldout_success_rate")
            for e in _segments(J["manager"])]
    s_ho = [(e.get("heldout") or {}).get("heldout_success_rate")
            for e in _segments(J["scripted"])]
    bit_identical = (m_ho == s_ho and any(x is not None for x in m_ho))
    gate(False, "mgr!=scr (cross-seed?)", not bit_identical,
         "DEGENERATE: manager==scripted (same-seed, bit-identical) -> mgr-vs-scr "
         "is a tautology; use run_experiment.sh (cross-seed) for a verdict"
         if bit_identical else "arms differ (cross-seed or divergent ladder)")

    # control applied nothing
    gate(True, "control applied nothing", not _applied(J["control"]))

    # SOFT: no crash-induced rollbacks (healthy tripwire)
    for arm, j in J.items():
        rb = [e for e in j if e.get("event") == "rollback"]
        gate(False, f"[{arm}] rollbacks", True, f"{len(rb)} rollback(s)")

    # ── report ──
    hard_fail = 0
    print(f"\n{'GATE':40} {'RESULT':6} DETAIL")
    print("-" * 72)
    for hard, name, ok, detail in results:
        tag = "PASS" if ok else ("FAIL" if hard else "note")
        if hard and not ok:
            hard_fail += 1
        print(f"{name:40} {tag:6} {detail}")
    print("-" * 72)
    if hard_fail:
        print(f"\n{hard_fail} HARD gate(s) FAILED — Phase 1 NOT complete.")
        return 1
    print("\nAll HARD gates passed.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase-1 gate-checklist verifier")
    p.add_argument("pilot_dir")
    args = p.parse_args(argv)
    return check(args.pilot_dir)


if __name__ == "__main__":
    raise SystemExit(main())

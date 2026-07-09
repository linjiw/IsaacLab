# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Analyze a completed Phase-1 pilot: the ON-vs-OFF comparison across the
control / manager / scripted arms on the PROTECTED held-out metric, with the
per-condition decomposition that guards against a single dominating command
(the SONIC-null lesson). Prints a plain-text report; no plotting deps.

Usage: analyze_pilot.py <pilot_dir>   (holds journal_{manager,scripted,control}.json)

CRUCIAL FRAMING (do not overclaim): a single-seed pilot is NOT evidence about
adaptivity. This report exists to (a) confirm the experiment ran cleanly and
(b) show the held-out trajectories side by side so the review can judge
trustworthiness — NOT to declare a winner."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _load(p):
    with open(p) as f:
        return json.load(f)


def _heldout_series(journal) -> List[Any]:
    return [(e.get("heldout") or {}).get("heldout_success_rate")
            for e in journal if "segment" in e]


def _final_heldout(journal):
    s = [h for h in _heldout_series(journal) if h is not None]
    return s[-1] if s else None


def _applied(journal):
    return [(e["tick"], e["decision"]["knob"], e["decision"]["value"])
            for e in journal if e.get("applied")]


def _per_condition_last(journal) -> Dict[str, float]:
    for e in reversed(journal):
        pc = (e.get("heldout") or {}).get("heldout_per_condition")
        if isinstance(pc, dict) and pc:
            return pc
    return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analyze a Phase-1 pilot")
    p.add_argument("pilot_dir")
    args = p.parse_args(argv)

    arms = {}
    for arm in ("control", "manager", "scripted"):
        path = os.path.join(args.pilot_dir, f"journal_{arm}.json")
        if os.path.exists(path):
            arms[arm] = _load(path)

    if not arms:
        print(f"no journals in {args.pilot_dir}")
        return 2

    print("=" * 72)
    print("PHASE-1 PILOT — ON-vs-OFF on the PROTECTED held-out metric")
    print("(single-seed pilot: a sanity/trustworthiness check, NOT an "
          "adaptivity claim)")
    print("=" * 72)

    print(f"\n{'arm':10} {'held-out trajectory (per segment)':42} {'final':>7} {'#moves':>7}")
    print("-" * 72)
    for arm, j in arms.items():
        series = _heldout_series(j)
        traj = " ".join(f"{h:.3f}" if h is not None else " -- " for h in series)
        fin = _final_heldout(j)
        fin_s = f"{fin:.3f}" if fin is not None else "n/a"
        print(f"{arm:10} {traj:42} {fin_s:>7} {len(_applied(j)):>7}")

    # decision ladders
    print("\n--- realized decision ladders (tick -> knob = value) ---")
    for arm, j in arms.items():
        moves = _applied(j)
        if moves:
            print(f"{arm:10} " + "; ".join(f"t{t}:{k}={v}" for t, k, v in moves))
        else:
            print(f"{arm:10} (no moves)")

    # manager vs scripted faithfulness
    if "manager" in arms and "scripted" in arms:
        m = {t: (k, v) for t, k, v in _applied(arms["manager"])}
        s = {t: (k, v) for t, k, v in _applied(arms["scripted"])}
        faithful = all(m.get(t) == v for t, v in s.items())
        print(f"\nscripted replays manager ladder faithfully: {faithful} "
              f"(manager {len(m)} moves, scripted {len(s)} moves)")

    # per-condition decomposition (final) — detect a dominating command
    print("\n--- final per-condition held-out success (dominating-condition check) ---")
    for arm, j in arms.items():
        pc = _per_condition_last(j)
        if pc:
            lo = min(pc.values()); hi = max(pc.values())
            print(f"{arm:10} spread [{lo:.3f}, {hi:.3f}] over {len(pc)} cmds; "
                  f"worst={min(pc, key=pc.get)} best={max(pc, key=pc.get)}")
        else:
            print(f"{arm:10} (no per-condition data)")

    # honest verdict scaffold
    print("\n--- comparison (held-out final) ---")
    fins = {a: _final_heldout(j) for a, j in arms.items()}
    base = fins.get("control")
    for arm in ("manager", "scripted"):
        if fins.get(arm) is not None and base is not None:
            d = fins[arm] - base
            print(f"  {arm} - control = {d:+.4f} held-out")
    if fins.get("manager") is not None and fins.get("scripted") is not None:
        d = fins["manager"] - fins["scripted"]
        print(f"  manager - scripted = {d:+.4f} held-out  "
              f"(the null-result question)")

    # ── equivalence-gate verdict (EVAL_FRAMEWORK §2) ──
    # apply the chaos-floor gate to the held-out TRAJECTORIES. tau MUST be
    # measured for THIS engine (a multi-seed chaos probe); until then we
    # report the gate with the SONIC tau as a PLACEHOLDER and say so loudly.
    print("\n--- equivalence-gate verdict (held-out trajectories) ---")
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "runmanager"))
        from core.equivalence import (E5B_CHAOS_FLOOR_MEAN, EquivalenceGate,
                                       calibrate_tau)
        tau = calibrate_tau(E5B_CHAOS_FLOOR_MEAN, safety_factor=3.0)
        print(f"  tau = {tau:.4f}  [PLACEHOLDER from SONIC E5B floor "
              f"{E5B_CHAOS_FLOOR_MEAN}; MUST re-measure for Isaac Lab/skrl "
              "via a 3-seed chaos probe before any real verdict]")
        gate = EquivalenceGate(tau)

        def _series(arm):
            return [h for h in _heldout_series(arms[arm]) if h is not None]
        for a, b in (("manager", "scripted"), ("manager", "control")):
            if a in arms and b in arms:
                sa, sb = _series(a), _series(b)
                n = min(len(sa), len(sb))
                if n >= 2:
                    rep = gate.compare(sa[:n], sb[:n])
                    print(f"  {a} vs {b}: {rep.verdict}  "
                          f"(mean_rel_dev={rep.mean_rel_dev}, tau={tau:.4f})")
                else:
                    print(f"  {a} vs {b}: too few points ({n})")
    except Exception as e:  # noqa: BLE001
        print(f"  (equivalence gate unavailable: {e})")

    print("\nNOTE: single-seed differences are almost certainly within "
          "run-to-run noise. Per EVAL_FRAMEWORK.md, a real verdict needs a "
          "MEASURED Isaac Lab noise band (3-seed chaos probe) + multi-seed "
          "arms. The pilot's claims are limited to: machinery runs clean, "
          "manager makes real gated decisions, held-out metric is protected "
          "+ discriminating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

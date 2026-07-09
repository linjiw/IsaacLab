# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Isaac Lab task catalog: which registered tasks are curriculum-RL ready.

Joins the declared tasks.yaml (gate metric + knob subset + framework) against
the knob registry, and reports which tasks can be driven by the run-manager
loop today. Pure Python + PyYAML (CPU) — enumerating the LIVE gym registry
needs Isaac Sim (use scripts/environments/list_envs.py for that)."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASKS = os.path.join(_HERE, "tasks.yaml")
DEFAULT_REGISTRY = os.path.normpath(
    os.path.join(_HERE, os.pardir, "isaaclab-knob-registry", "registry.yaml"))


def load_tasks(path: str = DEFAULT_TASKS) -> Dict[str, Any]:
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("tasks", {})


def load_registry_knob_names(path: str = DEFAULT_REGISTRY) -> List[str]:
    with open(path) as f:
        return sorted((yaml.safe_load(f) or {}).get("knobs", {}).keys())


def readiness(tasks: Dict[str, Any], knob_names: List[str]) -> Dict[str, Any]:
    """Per-task readiness report. A task is ready when it declares a
    gate_metric and a non-empty knob_subset whose every knob is in the
    registry (unknown knobs are a wiring bug, surfaced explicitly)."""
    known = set(knob_names)
    out: Dict[str, Any] = {}
    for task, spec in tasks.items():
        subset = spec.get("knob_subset") or []
        unknown = [k for k in subset if k not in known]
        has_gate = bool(spec.get("gate_metric"))
        ready = has_gate and bool(subset) and not unknown
        out[task] = {
            "ready": ready,
            "family": spec.get("family"),
            "frameworks": spec.get("frameworks", []),
            "gate_metric": (spec.get("gate_metric") or {}).get("kind"),
            "n_knobs": len(subset),
            "unknown_knobs": unknown,
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Isaac Lab curriculum-RL task catalog")
    p.add_argument("cmd", choices=["ready", "list"], default="ready", nargs="?")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    args = p.parse_args(argv)

    tasks = load_tasks(args.tasks)
    knobs = load_registry_knob_names(args.registry)
    if args.cmd == "list":
        # plain declaration dump (no registry join)
        for task, spec in tasks.items():
            print(f"{task:44} family={spec.get('family'):12} "
                  f"workflow={spec.get('workflow')} "
                  f"frameworks={spec.get('frameworks')}")
        return 0
    rep = readiness(tasks, knobs)
    for task, r in rep.items():
        mark = "READY" if r["ready"] else "  -- "
        extra = f" unknown={r['unknown_knobs']}" if r["unknown_knobs"] else ""
        print(f"[{mark}] {task:40} family={r['family']:12} "
              f"gate={r['gate_metric']} knobs={r['n_knobs']}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

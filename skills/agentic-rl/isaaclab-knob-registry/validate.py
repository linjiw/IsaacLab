# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Thin CLI over the vendored KnobRegistry validator.

Loads an Isaac Lab registry.yaml and either (a) reports the action space, or
(b) validates a single decision dict against a fresh RunState. Uses the
vendored runmanager/core/registry.py — no engine, no GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys

# reach the vendored core (skills/agentic-rl/runmanager)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNMANAGER = os.path.normpath(os.path.join(_HERE, os.pardir, "runmanager"))
sys.path.insert(0, _RUNMANAGER)

from core.registry import KnobRegistry, RunState  # noqa: E402

DEFAULT_REGISTRY = os.path.join(_HERE, "registry.yaml")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Isaac Lab knob registry validator")
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--decision", help="JSON decision dict to validate")
    p.add_argument("--tick", type=int, default=10)
    args = p.parse_args(argv)

    reg = KnobRegistry.load(
        args.registry,
        heldout_gated_families=_heldout_families(args.registry))
    if not args.decision:
        print(f"loaded {len(reg.knobs)} knobs from {args.registry}:")
        for name, k in sorted(reg.knobs.items()):
            rng = k.get("hard_range", k.get("choices"))
            print(f"  {name:32} family={k['family']:15} range={rng} "
                  f"step={k['max_step']} status={k['status']}")
        return 0

    decision = json.loads(args.decision)
    res = reg.validate_decision(decision, RunState(tick=args.tick))
    print(json.dumps({"ok": res.ok, "errors": res.errors,
                      "warnings": res.warnings}, indent=2))
    return 0 if res.ok else 1


def _heldout_families(path: str):
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    fams = (spec.get("meta") or {}).get("heldout_gated_families")
    return tuple(fams) if fams else ("schedule",)


if __name__ == "__main__":
    raise SystemExit(main())

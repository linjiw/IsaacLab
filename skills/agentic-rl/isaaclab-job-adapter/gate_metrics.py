# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Per-task-family gate-metric computation for eval_rollout.

Each gate takes the RAW per-env quantities (already pulled from the env by the
caller, which needs Isaac Lab) and returns a per-env boolean "within tolerance /
success" mask. Keeping the arithmetic here — separate from the Isaac Lab env
access — makes the gate logic PURE and CPU-testable (test_gate_metrics.py),
which matters because eval_rollout itself can only run in-container.

Two families implemented:
  - locomotion (tracking_within_tol): commanded vs actual base velocity L2 < tol
  - manipulation (lifted_and_goal_within_tol): object above minimal_height AND
    object-to-goal distance < tol  (Lift env: object_is_lifted + object_goal_distance)

`import numpy` is optional — the functions accept anything with elementwise
comparison + a `.sum()`/`bool` reduction (torch tensors in-container, numpy or
plain floats in tests), so this module has NO hard torch/numpy dependency.
"""

from __future__ import annotations

from typing import Any, Sequence


def _l2_rows(a, b):
    """Row-wise L2 norm of (a - b) for two (N, K) nested sequences. Pure Python
    fallback so this is testable without numpy/torch; in-container the caller
    passes torch tensors and uses the tensor path (compute_* below accept both
    via duck-typed ops, but tests use this list path)."""
    out = []
    for ra, rb in zip(a, b):
        out.append(sum((x - y) ** 2 for x, y in zip(ra, rb)) ** 0.5)
    return out


def tracking_within_tol(commanded: Sequence[Sequence[float]],
                        actual: Sequence[Sequence[float]],
                        tol: float) -> list:
    """Locomotion gate: per-env, is the tracked command within `tol` (L2)?

    commanded/actual: (N, 3) [lin_vel_x, lin_vel_y, ang_vel_z]. Returns a list
    of bools (True = within tolerance = a 'hit')."""
    return [d < tol for d in _l2_rows(commanded, actual)]


def lifted_and_goal_within_tol(object_z: Sequence[float],
                               object_pos: Sequence[Sequence[float]],
                               goal_pos: Sequence[Sequence[float]],
                               minimal_height: float,
                               tol: float) -> list:
    """Manipulation gate (Lift): per-env, is the object BOTH lifted above
    `minimal_height` AND within `tol` (L2) of the goal position?

    object_z: (N,) object world-frame height. object_pos/goal_pos: (N, 3) world
    positions. Mirrors object_is_lifted * (object_goal_distance < tol) from the
    Lift env's reward terms, but as a hard success mask (not a tanh reward)."""
    dists = _l2_rows(object_pos, goal_pos)
    return [(z > minimal_height) and (d < tol)
            for z, d in zip(object_z, dists)]


def success_fraction(mask: Sequence[Any]) -> float:
    """Fraction of True in a per-env success mask (the gate scalar for one
    step); averaged over steps by the caller -> heldout_success_rate."""
    m = list(mask)
    return (sum(1 for x in m if x) / len(m)) if m else 0.0


# registry of gate kinds -> which raw quantities they need, for eval_rollout to
# dispatch on tasks.yaml gate_metric.kind. (Documentation + a guard against
# an unknown kind.)
GATE_KINDS = {
    "tracking_within_tol": ("commanded", "actual", "tol"),
    "lifted_and_goal_within_tol": ("object_z", "object_pos", "goal_pos",
                                   "minimal_height", "tol"),
}

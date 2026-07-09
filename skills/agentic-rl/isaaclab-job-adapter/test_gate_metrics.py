# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""CPU tests for the pure gate-metric arithmetic (no Isaac Lab / GPU)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate_metrics import (  # noqa: E402
    GATE_KINDS,
    lifted_and_goal_within_tol,
    success_fraction,
    tracking_within_tol,
)


# ── locomotion ────────────────────────────────────────────────────────
def test_tracking_within_tol_basic():
    # env0 tracks exactly (dist 0 < 0.25 -> hit); env1 off by 0.5 -> miss
    cmd = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    act = [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    assert tracking_within_tol(cmd, act, tol=0.25) == [True, False]


def test_tracking_boundary():
    # exactly at tol is NOT within (strict <)
    assert tracking_within_tol([[0.25, 0, 0]], [[0.0, 0, 0]], tol=0.25) == [False]
    assert tracking_within_tol([[0.24, 0, 0]], [[0.0, 0, 0]], tol=0.25) == [True]


# ── manipulation ──────────────────────────────────────────────────────
def test_lifted_and_goal_both_required():
    # env0: lifted + at goal -> hit; env1: at goal but NOT lifted -> miss;
    # env2: lifted but far from goal -> miss; env3: lifted + near goal -> hit
    z = [0.10, 0.02, 0.10, 0.10]
    obj = [[0, 0, 0.10], [0, 0, 0.02], [0, 0, 0.10], [0.01, 0, 0.10]]
    goal = [[0, 0, 0.10], [0, 0, 0.02], [1.0, 0, 0.10], [0.0, 0, 0.10]]
    out = lifted_and_goal_within_tol(z, obj, goal, minimal_height=0.04, tol=0.05)
    assert out == [True, False, False, True]


def test_lifted_gate_height_boundary():
    # z exactly at minimal_height is NOT lifted (strict >)
    z = [0.04]
    obj = [[0, 0, 0.04]]
    goal = [[0, 0, 0.04]]
    assert lifted_and_goal_within_tol(z, obj, goal, 0.04, 0.05) == [False]


# ── success fraction ──────────────────────────────────────────────────
def test_success_fraction():
    assert success_fraction([True, False, True, True]) == 0.75
    assert success_fraction([]) == 0.0
    assert success_fraction([False, False]) == 0.0


def test_gate_kinds_registry_documents_inputs():
    assert set(GATE_KINDS) == {"tracking_within_tol", "lifted_and_goal_within_tol"}
    assert "object_z" in GATE_KINDS["lifted_and_goal_within_tol"]
    assert "commanded" in GATE_KINDS["tracking_within_tol"]

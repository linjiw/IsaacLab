# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""CPU tests for the Isaac Lab held-out watcher: command-grid keys, salted
split determinism + growth stability, manifest integrity, and the
refuse-on-foreign-keys guard on the protected metric."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from holdout import (  # noqa: E402
    command_grid,
    command_key,
    curriculum_keys,
    heldout_commands,
    heldout_record_from_metrics_eval,
    load_manifest,
    parse_command_key,
    select_holdout,
    write_manifest,
)


# ── command grid / keys ──────────────────────────────────────────────
def test_command_key_roundtrip():
    k = command_key(1.5, 0.0, -2.0)
    assert parse_command_key(k) == (1.5, 0.0, -2.0)


def test_command_grid_is_cartesian_and_sorted_unique():
    g = command_grid([-1, 1], [-2, 2], lin_vel_y=[0])
    assert len(g) == 4
    assert g == sorted(g)
    assert len(set(g)) == len(g)


# ── salted split ─────────────────────────────────────────────────────
def test_split_deterministic_same_salt():
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    a = select_holdout(keys, 0.34, "saltA")
    b = select_holdout(keys, 0.34, "saltA")
    assert a == b


def test_split_changes_with_salt():
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    a = set(select_holdout(keys, 0.34, "saltA")["heldout"])
    b = set(select_holdout(keys, 0.34, "saltB")["heldout"])
    assert a != b


def test_split_partitions_completely():
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    s = select_holdout(keys, 0.34, "x")
    assert set(s["heldout"]) | set(s["curriculum"]) == set(keys)
    assert not (set(s["heldout"]) & set(s["curriculum"]))


def test_split_stable_under_grid_growth():
    # a key's side must not change when the grid grows (hash-based, not index)
    small = command_grid([-2, -1, 1, 2], [-1, 1])
    big = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    s_small = select_holdout(small, 0.34, "grow")
    s_big = select_holdout(big, 0.34, "grow")
    for k in small:
        side_small = "heldout" if k in s_small["heldout"] else "curriculum"
        side_big = "heldout" if k in s_big["heldout"] else "curriculum"
        assert side_small == side_big, f"{k} moved sides on growth"


def test_bad_fraction_and_empty_salt_rejected():
    keys = command_grid([-1, 1], [-1, 1])
    with pytest.raises(ValueError):
        select_holdout(keys, 0.0, "s")
    with pytest.raises(ValueError):
        select_holdout(keys, 0.34, "")


def test_duplicate_key_rejected():
    with pytest.raises(ValueError):
        select_holdout(["k", "k"], 0.34, "s")


# ── manifest ─────────────────────────────────────────────────────────
def test_manifest_roundtrip_and_integrity(tmp_path):
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    path = str(tmp_path / "m.json")
    m = write_manifest(path, keys, 0.34, "salt42", tolerance=0.25)
    loaded = load_manifest(path)
    assert loaded["heldout_keys"] == m["heldout_keys"]
    assert loaded["tolerance"] == 0.25


def test_manifest_tamper_detected(tmp_path):
    import json
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    path = str(tmp_path / "m.json")
    write_manifest(path, keys, 0.34, "salt42")
    m = json.load(open(path))
    m["heldout_keys"].append("vx=9.0,vy=0.0,wz=9.0")  # sneak a key in
    json.dump(m, open(path, "w"))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_curriculum_keys_excludes_heldout(tmp_path):
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    path = str(tmp_path / "m.json")
    m = write_manifest(path, keys, 0.34, "salt42")
    curr = curriculum_keys(m, keys)
    assert not (set(curr) & set(m["heldout_keys"]))
    assert set(curr) | set(m["heldout_keys"]) == set(keys)


def test_heldout_commands_are_tuples(tmp_path):
    keys = command_grid([-2, -1, 1, 2], [-1, 1])
    path = str(tmp_path / "m.json")
    m = write_manifest(path, keys, 0.34, "s")
    cmds = heldout_commands(m)
    assert all(len(c) == 3 for c in cmds)
    assert len(cmds) == len(m["heldout_keys"])


# ── protected-record guard ───────────────────────────────────────────
def _manifest(tmp_path):
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    path = str(tmp_path / "m.json")
    return write_manifest(path, keys, 0.34, "salt42")


def test_record_ok_on_clean_eval(tmp_path):
    m = _manifest(tmp_path)
    metrics = {"heldout_success_rate": 0.6, "eval_keys": m["heldout_keys"][:2],
               "mean_tracking_error": 0.4}
    rec = heldout_record_from_metrics_eval(metrics, m, it=100)
    assert rec["heldout_success_rate"] == 0.6
    assert rec["heldout_n_commands"] == len(m["heldout_keys"])
    assert rec["heldout_manifest_integrity"] == m["integrity"]


def test_record_refuses_foreign_keys(tmp_path):
    m = _manifest(tmp_path)
    metrics = {"heldout_success_rate": 0.6,
               "eval_keys": ["vx=99.0,vy=0.0,wz=0.0"]}  # not in manifest
    with pytest.raises(ValueError):
        heldout_record_from_metrics_eval(metrics, m, it=100)


def test_record_refuses_out_of_range(tmp_path):
    m = _manifest(tmp_path)
    with pytest.raises(ValueError):
        heldout_record_from_metrics_eval({"heldout_success_rate": 1.5}, m, it=1)


def test_record_falls_back_to_success_rate_name(tmp_path):
    m = _manifest(tmp_path)
    rec = heldout_record_from_metrics_eval({"success_rate": 0.5}, m, it=1)
    assert rec["heldout_success_rate"] == 0.5

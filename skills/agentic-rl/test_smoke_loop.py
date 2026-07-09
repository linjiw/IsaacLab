# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Phase-0/1 smoke: the full run-manager loop composes over the Isaac Lab
registry + Isaac Lab policies with a CPU mock adapter — all three arms
(control / manager / scripted) — now with the PROTECTED held-out metric and
a scripted ladder DERIVED from the manager's realized journal. No GPU.

This is the ON-vs-OFF harness in miniature, faithful to the SONIC methodology:
- the manager gates on the held-out metric (not the signal it steers),
- the scripted arm replays the manager's exact realized decisions,
- all three share the identical guardrailed loop."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "isaaclab-heldout-watcher"))

import pytest  # noqa: E402

import run_curriculum as rc  # noqa: E402
from holdout import command_grid, write_manifest  # noqa: E402
from isaaclab_policies import ladder_from_journal  # noqa: E402


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory):
    """A real held-out manifest so the loop's heldout_hook (which loads +
    integrity-checks it) exercises the true code path, not a stub."""
    path = str(tmp_path_factory.mktemp("heldout") / "manifest.json")
    keys = command_grid([-3, -2, -1, 1, 2, 3], [-2, -1, 1, 2])
    write_manifest(path, keys, 0.34, "smoke-salt", tolerance=0.25)
    return path


class _Args:
    def __init__(self, arm, manifest="/mock/manifest.json", **kw):
        self.arm = arm
        self.segments = 8
        self.iterations = 30
        self.task = "Isaac-Velocity-Rough-Anymal-C-v0"
        self.framework = "skrl"
        self.num_envs = None
        self.eval_envs = 64
        self.gate_metric = "tracking"
        self.seed = 42
        self.no_eval = False
        self.dry_run = True
        self.journal_out = None
        self.warm_start = "/mock/warm_start.pt"
        self.heldout_manifest = manifest   # truthy → held-out path
        self.scripted_from_journal = None
        self.t_low = 0.50
        self.t_high = 0.85
        self.sustain = 3
        self.base_knob = []
        for k, v in kw.items():
            setattr(self, k, v)


def _run(arm, manifest_path=None, adapter=None, **kw):
    from core.loop import RunManager
    args = _Args(arm, manifest=manifest_path, **kw) if manifest_path else _Args(arm, **kw)
    registry = rc.load_registry()
    adapter = adapter or rc.make_adapter(args)   # default gate >= t_high → hardens
    policy = rc.make_policy(arm, args)
    config = rc.make_config(arm, args, adapter)
    mgr = RunManager(policy=policy, adapter=adapter, registry=registry, config=config)
    summary = mgr.run(args.segments)
    return mgr, summary


def _mock(gate_script, **kw):
    return rc.MockLocomotionAdapter(gate_script=gate_script, heldout=True, **kw)


def test_control_arm_applies_nothing(manifest_path):
    mgr, s = _run("control", manifest_path=manifest_path)
    assert s["decisions_applied"] == 0
    assert s["rollbacks"] == 0
    assert len([e for e in mgr.journal if "segment" in e]) == 8


def test_manager_gates_on_heldout_and_hardens(manifest_path):
    mgr, s = _run("manager", manifest_path=manifest_path)
    # held-out series populated for every segment (protected metric wired)
    heldout = [(e.get("heldout") or {}).get("heldout_success_rate")
               for e in mgr.journal if "segment" in e]
    assert all(h is not None for h in heldout), "held-out metric not populated"
    # gate held >= t_high → the manager widens the command range, gated
    # one-change-at-a-time
    assert s["decisions_applied"] >= 2
    applied = [e for e in mgr.journal if e.get("applied")]
    assert all(e["decision"]["knob"] == "command_range_lin_vel_x" for e in applied)
    vals = [e["decision"]["value"] for e in applied]
    assert vals == sorted(vals)  # monotonically hardening


def test_manager_refuses_without_heldout():
    # no held-out manifest → hard rule 4 → the manager proposes nothing
    mgr, s = _run("manager", heldout_manifest=None)
    assert s["decisions_applied"] == 0
    reasons = [(e.get("decision") or {}).get("reason", "") for e in mgr.journal]
    assert any("hard rule 4" in r for r in reasons)


def test_manager_eases_under_low_heldout(manifest_path):
    # held-out pinned <= t_low → the manager NARROWS the command range (easier)
    mgr, s = _run("manager", manifest_path=manifest_path,
                  adapter=_mock([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    applied = [e for e in mgr.journal if e.get("applied")]
    assert applied, "manager made no EASE move under a low held-out gate"
    assert all(e["decision"]["knob"] == "command_range_lin_vel_x" for e in applied)
    vals = [e["decision"]["value"] for e in applied]
    assert vals == sorted(vals, reverse=True)  # monotonically narrowing (easing)


def test_manager_anneals_lr_on_in_band_plateau(manifest_path):
    # held-out in-band + FLAT reward → the manager anneals learning_rate
    mgr, s = _run("manager", manifest_path=manifest_path,
                  adapter=_mock([0.7, 0.7, 0.7, 0.7, 0.7, 0.7], flat_reward=True))
    applied = [e for e in mgr.journal if e.get("applied")]
    assert any(e["decision"]["knob"] == "learning_rate" for e in applied), \
        "manager did not anneal lr on an in-band plateau"
    lr_moves = [e for e in applied if e["decision"]["knob"] == "learning_rate"]
    # the lr-anneal effect check must be meaningful (held-out, not vacuous)
    chk = lr_moves[0]["decision"].get("expected_effect_check")
    assert chk and chk["metric"] == "eval/heldout_success_rate"


def test_scripted_replays_manager_journal(manifest_path):
    # 1) run the manager, capture its realized journal
    mgr_m, _ = _run("manager", manifest_path=manifest_path)
    ladder = ladder_from_journal(mgr_m.journal)
    assert ladder, "manager made no applied decisions to replay"
    # 2) run scripted from that ladder; it must reproduce the same knob/values
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(mgr_m.journal, f)
        jpath = f.name
    mgr_s, s = _run("scripted", manifest_path=manifest_path,
                    scripted_from_journal=jpath)
    os.unlink(jpath)
    applied_s = [(e["tick"], e["decision"]["knob"], e["decision"]["value"])
                 for e in mgr_s.journal if e.get("applied")]
    # every scripted move must be a rung the manager actually walked
    for tick, knob, val in applied_s:
        assert ladder.get(tick) == (knob, val)


def test_scripted_arm_requires_a_journal(manifest_path):
    with pytest.raises(SystemExit):
        _run("scripted", manifest_path=manifest_path, scripted_from_journal=None)


def test_all_arms_share_the_same_guardrailed_loop(manifest_path):
    for arm in ("control", "manager"):
        mgr, s = _run(arm, manifest_path=manifest_path)
        assert s["arm"] == arm
        assert s["rollbacks"] == 0  # healthy gate → nothing reverts


def test_ladder_from_journal_extracts_only_applied_sets():
    journal = [
        {"tick": 1, "decision": {"action": "none"}},
        {"tick": 2, "decision": {"action": "set", "knob": "command_range_lin_vel_x",
                                 "value": 1.25}, "applied": True},
        {"tick": 3, "decision": {"action": "set", "knob": "learning_rate",
                                 "value": 0.0006}},  # not applied (rejected)
    ]
    ladder = ladder_from_journal(journal)
    assert ladder == {2: ("command_range_lin_vel_x", 1.25)}

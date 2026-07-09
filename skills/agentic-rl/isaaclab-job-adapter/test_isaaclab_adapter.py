# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""CPU tests for the pure surface of the Isaac Lab job adapter: knob→Hydra
mapping, train/eval command building, TB-scalar normalization, console
parsing, and metrics parsing. No container / GPU needed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from isaaclab_adapter import (  # noqa: E402
    CONTAINER,
    ISAACLAB_DIR,
    _safe_load_isaac_yaml,
    build_eval_command,
    build_overrides,
    build_train_command,
    knob_to_config_path,
    knob_to_hydra,
    normalize_tag,
    parse_console_log,
    parse_metrics_eval,
    records_from_scalars,
)


# ── knob mapping ─────────────────────────────────────────────────────
def test_knob_map_framework_specific_kl_path():
    # skrl KL target is nested under the scheduler kwargs (verified)
    assert (knob_to_hydra("skrl")["desired_kl"]
            == "agent.agent.learning_rate_scheduler_kwargs.kl_threshold")
    assert (knob_to_hydra("rl_games")["desired_kl"]
            == "agent.params.config.kl_threshold")


def test_env_knobs_are_framework_independent():
    assert (knob_to_hydra("skrl")["command_range_lin_vel_x"]
            == knob_to_hydra("rl_games")["command_range_lin_vel_x"]
            == "env.commands.base_velocity.ranges.lin_vel_x")


def test_build_overrides_raises_on_unmapped_knob():
    with pytest.raises(KeyError):
        build_overrides({"not_a_knob": 1.0}, "skrl")


def test_build_overrides_sorted_and_formatted():
    ov = build_overrides({"learning_rate": 8e-4,
                          "command_range_lin_vel_x": 2.0}, "skrl")
    # sorted by knob name: command_range_* before learning_rate
    # range knob: scalar half-width 2.0 -> symmetric SPACE-FREE "[-2.0,2.0]"
    assert ov[0] == "env.commands.base_velocity.ranges.lin_vel_x=[-2.0,2.0]"
    assert ov[1] == "agent.agent.learning_rate=0.0008"


def test_range_knob_expands_symmetric():
    ov = build_overrides({"command_range_ang_vel_z": 1.5}, "skrl")
    assert ov == ["env.commands.base_velocity.ranges.ang_vel_z=[-1.5,1.5]"]


def test_range_override_has_no_space_survives_word_split():
    # review B2: a space in the list token would be word-split by
    # `docker exec bash -c` and abort the launch. Assert no whitespace.
    ov = build_overrides({"command_range_lin_vel_x": 1.5}, "skrl")[0]
    assert " " not in ov and "\t" not in ov


def test_train_command_range_knob_reaches_python_as_one_arg():
    # the built inner shell string must, when word-split by bash, keep the
    # range override as a single token (shlex.quote guards it).
    import shlex as _shlex
    cmd = build_train_command(
        "skrl", "Isaac-Velocity-Rough-Anymal-C-v0", experiment_name="seg1",
        iterations=30, knobs={"command_range_lin_vel_x": 1.5})
    inner = cmd[-1]
    tokens = _shlex.split(inner)   # exactly how bash -c would split it
    assert "env.commands.base_velocity.ranges.lin_vel_x=[-1.5,1.5]" in tokens


def test_config_path_excludes_range_knobs_but_keeps_scalars():
    # range knobs omitted (scalar belief can't match a resolved tuple)
    cp = knob_to_config_path("rl_games")
    assert "command_range_lin_vel_x" not in cp
    assert cp["learning_rate"] == "params.config.learning_rate"


def test_bad_framework_rejected():
    with pytest.raises(ValueError):
        knob_to_hydra("rsl_rl")


# ── train command ────────────────────────────────────────────────────
def test_train_command_shape_skrl():
    cmd = build_train_command(
        "skrl", "Isaac-Velocity-Rough-Anymal-C-v0", experiment_name="seg1",
        iterations=50, knobs={"learning_rate": 8e-4}, checkpoint="/x/snap.pt",
        num_envs=1024, seed=7)
    assert cmd[:5] == ["docker", "exec", CONTAINER, "bash", "-c"]
    inner = cmd[-1]
    assert f"cd {ISAACLAB_DIR} &&" in inner
    assert "/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py" in inner
    assert "--task=Isaac-Velocity-Rough-Anymal-C-v0" in inner
    assert "--headless" in inner
    assert "--max_iterations=50" in inner
    assert "--seed=7" in inner
    assert "--num_envs=1024" in inner
    assert "--checkpoint=/x/snap.pt" in inner
    assert "--algorithm=PPO" in inner and "--ml_framework=torch" in inner
    assert "agent.agent.experiment.experiment_name=seg1" in inner
    assert "agent.agent.learning_rate=0.0008" in inner


def test_train_command_rl_games_experiment_override():
    cmd = build_train_command(
        "rl_games", "Isaac-Velocity-Rough-Anymal-C-v0", experiment_name="seg2",
        iterations=10, knobs={})
    inner = cmd[-1]
    assert "agent.params.config.full_experiment_name=seg2" in inner
    # rl_games has no --algorithm/--ml_framework flags
    assert "--algorithm" not in inner


def test_train_command_no_checkpoint_omits_flag():
    cmd = build_train_command("skrl", "T-v0", experiment_name="s", iterations=5)
    assert "--checkpoint" not in cmd[-1]


def test_train_command_background_redirect():
    cmd = build_train_command("skrl", "T-v0", experiment_name="s", iterations=5,
                              log_path="/logs/s.log")
    inner = cmd[-1]
    assert inner.startswith(f"cd {ISAACLAB_DIR} && nohup")
    assert inner.rstrip().endswith("&")
    assert "> /logs/s.log 2>&1 &" in inner


# ── eval command ─────────────────────────────────────────────────────
def test_eval_command_uses_bundled_script_and_out_dir():
    cmd = build_eval_command("skrl", "Isaac-Velocity-Rough-Anymal-C-Play-v0",
                             "/x/snap.pt", "/tmp/eval_out", num_envs=64,
                             gate_metric="tracking")
    inner = cmd[-1]
    assert "eval_rollout.py" in inner
    assert "--out_dir=/tmp/eval_out" in inner
    assert "--num_envs=64" in inner
    assert "--gate_metric=tracking" in inner
    # eval must NOT carry manager knob overrides (pin-the-scoreboard)
    assert "agent.agent.learning_rate" not in inner


def test_eval_command_pins_passed_through():
    cmd = build_eval_command("skrl", "T-Play-v0", "/x.pt", "/o",
                             extra_overrides=["env.rewards.action_rate_l2.weight=-0.01"])
    assert "env.rewards.action_rate_l2.weight=-0.01" in cmd[-1]


# ── scalar normalization ─────────────────────────────────────────────
def test_normalize_tag_aliases_and_passthrough():
    assert normalize_tag("rewards/step") == "Episode/rew_mean"
    assert normalize_tag("Reward / Total reward (mean)") == "Episode/rew_mean"
    assert normalize_tag("Episode_Termination/base_contact") == "Episode_Termination/base_contact"
    assert normalize_tag("Curriculum/terrain_levels") == "Curriculum/terrain_levels"
    assert normalize_tag("some/unmapped/tag") is None


def test_normalize_real_skrl_tags():
    # VERIFIED against a real Isaac-Cartpole-v0 skrl run's TB tags (2026-07-08)
    assert normalize_tag("Episode / Total timesteps (mean)") == "Episode/len_mean"
    assert normalize_tag("Reward / Total reward (mean)") == "Episode/rew_mean"
    assert normalize_tag("Episode_Reward/alive") == "Episode_Reward/alive"
    # diagnostics we intentionally drop
    assert normalize_tag("Loss / Policy loss") is None
    assert normalize_tag("Stats / Inference time (ms)") is None


def test_records_from_scalars_groups_by_step():
    scalars = {
        "rewards/step": [(0, 1.0), (1, 2.0)],
        "Episode_Termination/base_contact": [(0, 0.3), (1, 0.2)],
        "junk/tag": [(0, 99.0)],
    }
    recs = records_from_scalars(scalars)
    assert [r["it"] for r in recs] == [0, 1]
    assert recs[0]["Episode/rew_mean"] == 1.0
    assert recs[0]["Episode_Termination/base_contact"] == 0.3
    assert "junk/tag" not in recs[0]
    assert recs[1]["Episode/rew_mean"] == 2.0


def test_records_from_scalars_empty():
    assert records_from_scalars({}) == []


# ── console parsing ──────────────────────────────────────────────────
def test_parse_console_extracts_experiment_and_logroot():
    lines = [
        "[INFO] Logging experiment in directory: /workspace/isaaclab/logs/skrl/anymal_c_rough_direct",
        "Exact experiment name requested from command line: seg1",
        "... training ...",
    ]
    p = parse_console_log(lines)
    assert p.log_root == "/workspace/isaaclab/logs/skrl/anymal_c_rough_direct"
    assert p.experiment_name == "seg1"
    assert p.tracebacks == 0


def test_parse_console_counts_tracebacks():
    lines = ["ok", "Traceback (most recent call last):", "  File ...", "Error"]
    assert parse_console_log(lines).tracebacks == 1


# ── metrics parsing ──────────────────────────────────────────────────
def test_parse_metrics_eval_maps_gate_to_success_rate():
    metrics = {"tracking": 0.72, "mean_reward": 12.3,
               "failed_keys": ["cmd_hi_speed"]}
    rec = parse_metrics_eval(metrics, it=100, gate_metric="tracking")
    assert rec["it"] == 100
    assert rec["tracking"] == 0.72
    assert rec["success_rate"] == 0.72   # gate copied to success_rate
    assert rec["mean_reward"] == 12.3
    assert rec["failed_keys"] == ["cmd_hi_speed"]


def test_parse_metrics_eval_no_gate_present():
    rec = parse_metrics_eval({"mean_reward": 1.0}, it=5, gate_metric="tracking")
    assert "success_rate" not in rec
    assert rec["mean_reward"] == 1.0


def test_parse_metrics_eval_passes_per_condition_through():
    # R2: the per-condition dict must survive (previously non-scalars dropped)
    metrics = {"tracking": 0.6,
               "per_condition": {"vx=1.0,vy=0.0,wz=0.0": 0.9,
                                 "vx=3.0,vy=0.0,wz=2.0": 0.1}}
    rec = parse_metrics_eval(metrics, it=10, gate_metric="tracking")
    assert rec["per_condition"]["vx=3.0,vy=0.0,wz=2.0"] == 0.1


# ── resolved-config YAML with Isaac Lab's !!python/* tags ─────────────
def test_safe_load_isaac_yaml_handles_python_tags():
    # Isaac Lab's dump_yaml emits these tags; plain yaml.safe_load REFUSES
    # them. The custom loader must accept them (tuple->list, object->plain)
    # AND the result must round-trip through safe_dump/safe_load (the loop
    # does exactly that in _verify_config).
    import yaml
    doc = """
commands:
  base_velocity:
    ranges:
      lin_vel_x: !!python/tuple
      - -1.0
      - 1.0
    heading_command: true
scene:
  height_scanner:
    pattern_cfg: !!python/object:isaaclab.sensors.RayCasterCfg.PatternCfg
      resolution: 0.1
      ordering: xy
rewards:
  action_rate_l2:
    func: !!python/name:isaaclab.envs.mdp.action_rate_l2
    weight: -0.01
"""
    loaded = _safe_load_isaac_yaml(doc)
    assert loaded["commands"]["base_velocity"]["ranges"]["lin_vel_x"] == [-1.0, 1.0]
    assert loaded["rewards"]["action_rate_l2"]["weight"] == -0.01
    # the object tag became a plain mapping (no code executed)
    assert isinstance(loaded["scene"]["height_scanner"]["pattern_cfg"], dict)
    # round-trips through the plain safe loader the run-manager loop uses
    reloaded = yaml.safe_load(yaml.safe_dump(loaded))
    assert reloaded["rewards"]["action_rate_l2"]["weight"] == -0.01

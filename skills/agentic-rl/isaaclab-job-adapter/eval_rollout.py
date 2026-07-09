# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless eval-rollout that writes a metrics_eval.json scoreboard.

Isaac Lab's stock play.py loops forever for visualization and writes no
metrics file, so the curriculum-manager loop cannot use it as a scoreboard.
This script loads a checkpoint, runs a FIXED-LENGTH deterministic rollout on
the task's -Play-v0 variant (reduced envs, noise/DR disabled — the pinned
scoreboard), aggregates a scalar gate metric plus diagnostics, and writes
<out_dir>/metrics_eval.json.

Gate metric (locomotion default, kind=tracking_within_tol): the fraction of
env-steps whose commanded base velocity is tracked within `--gate_tol`
(L2 over the tracked axes). This is the DERIVED gate metric the task catalog
declares for locomotion tasks, which log tracking error rather than a native
success rate. Manipulation/dexterous tasks should pass --gate_metric matching
a term the env already logps as a success/progress signal.

Runs INSIDE the isaac-lab-base container (invoked by isaaclab_adapter's
build_eval_command via docker exec + /isaac-sim/python.sh).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Headless eval rollout → metrics_eval.json")
parser.add_argument("--framework", type=str, default="skrl", choices=["skrl", "rl_games"])
parser.add_argument("--task", type=str, required=True, help="the -Play-v0 task id")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_steps", type=int, default=1000, help="rollout length in env steps")
parser.add_argument("--seed", type=int, default=1234, help="fixed eval seed (pinned scoreboard)")
parser.add_argument("--gate_metric", type=str, default="tracking",
                    help="name recorded as the loop gate metric")
parser.add_argument("--gate_tol", type=float, default=0.25,
                    help="tracking tolerance (m/s or rad/s L2) for tracking_within_tol")
parser.add_argument("--heldout_manifest", type=str, default=None,
                    help="if set, run the PROTECTED held-out eval: FORCE the "
                         "manifest's frozen command grid (ignoring the env's "
                         "own command sampling) and report heldout_success_rate")
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# eval is always headless; clear argv for Hydra (env/agent overrides pass through)
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

if args_cli.framework == "skrl":
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    else:
        from skrl.utils.runner.jax import Runner
    _agent_entry = ("skrl_cfg_entry_point" if args_cli.algorithm.lower() == "ppo"
                    else f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point")
else:
    _agent_entry = "rl_games_cfg_entry_point"


def _velocity_command_term(env):
    """The UniformVelocityCommand term, or None if the env has no such command."""
    try:
        return env.unwrapped.command_manager.get_term("base_velocity")
    except Exception:  # noqa: BLE001 — not a velocity-command env
        return None


def _tracked_command_error(env) -> "torch.Tensor | None":
    """Per-env L2 tracking error between commanded and actual base velocity
    (lin_vel_x, lin_vel_y, ang_vel_z) for manager-based locomotion envs.
    Returns None if the env exposes no such command (non-locomotion task)."""
    unwrapped = env.unwrapped
    try:
        cmd = unwrapped.command_manager.get_command("base_velocity")  # (N, >=3)
    except Exception:  # noqa: BLE001 — not a velocity-command env
        return None
    robot = unwrapped.scene["robot"]
    lin = robot.data.root_lin_vel_b[:, :2]        # x, y in base frame
    ang = robot.data.root_ang_vel_b[:, 2:3]       # yaw rate
    actual = torch.cat([lin, ang], dim=1)
    return torch.norm(cmd[:, :3] - actual, dim=1)


def _force_commands(term, forced: "torch.Tensor") -> None:
    """Overwrite the velocity command term's buffer with `forced` (N,3).

    Called every step so the env's own periodic resampling / standing-env
    zeroing can never change the FROZEN held-out commands — the manager's
    training command-range knob thus cannot touch the protected metric."""
    term.vel_command_b[:, :3] = forced


@hydra_task_config(args_cli.task, _agent_entry)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # HELD-OUT metric validity (fix-verify CRITICAL): the env recomputes the
    # command BEFORE the observation each step (manager_based_rl_env.step:232
    # then :238), and UniformVelocityCommand._update_command OVERWRITES the yaw
    # command from a heading target (heading_command=True, rel_heading_envs=1.0
    # on the -Play cfg) and zeroes standing envs. If we only re-force AFTER
    # env.step, the policy OBSERVES the heading-derived yaw but is GRADED
    # against our forced yaw -> heldout_success_rate pinned ~0 regardless of
    # policy quality. Disable the overwriting mechanisms at the cfg level BEFORE
    # gym.make so the forced command is what the policy both sees and is graded
    # on. (Belt-and-suspenders: we still re-force after each step below.)
    if args_cli.heldout_manifest:
        try:
            cmd_cfg = env_cfg.commands.base_velocity
            cmd_cfg.heading_command = False
            cmd_cfg.rel_heading_envs = 0.0
            cmd_cfg.rel_standing_envs = 0.0
            # never resample within the eval horizon
            cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
        except AttributeError as e:
            raise SystemExit(
                f"--heldout_manifest: cannot pin the velocity command cfg "
                f"({e}); this task's command term is not the expected "
                "UniformVelocityCommand")
    env_cfg.seed = args_cli.seed
    if isinstance(agent_cfg, dict) and "seed" in agent_cfg:
        agent_cfg["seed"] = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm.lower() == "ppo":
        env = multi_agent_to_single_agent(env)

    resume_path = os.path.abspath(args_cli.checkpoint)

    if args_cli.framework == "skrl":
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, agent_cfg)
        runner.agent.load(resume_path)
        # skrl 2.1.0: set_running_mode was removed; enable_training_mode(False)
        # puts models in eval mode (also disables the value-head branch).
        runner.agent.enable_training_mode(False, apply_to_models=True)

        def policy(obs, states):
            # act() signature in 2.1.0 is act(obs, states, *, timestep, timesteps)
            # -> (actions, outputs_dict); the deterministic action is the
            # policy mean when present, else the sampled action (play.py:225-231).
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            return outputs[-1].get("mean_actions", outputs[0])
    else:
        raise SystemExit("rl_games eval path is a Phase-3 TODO; use skrl for Phase 1")

    # ── held-out setup: FORCE the manifest's frozen command grid ───
    forced = None
    heldout_keys = None
    cmd_term = _velocity_command_term(env)
    tol = args_cli.gate_tol
    if args_cli.heldout_manifest:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir,
            "isaaclab-heldout-watcher"))
        from holdout import heldout_commands, load_manifest  # noqa: E402
        manifest = load_manifest(args_cli.heldout_manifest)
        tol = float(manifest.get("tolerance", tol))
        cmds = heldout_commands(manifest)  # list of (vx,vy,wz)
        heldout_keys = list(manifest["heldout_keys"])
        if cmd_term is None:
            raise SystemExit("--heldout_manifest given but task has no "
                             "base_velocity command term")
        device = cmd_term.vel_command_b.device
        n = cmd_term.vel_command_b.shape[0]
        # assign the held-out commands round-robin across the N envs, so every
        # forced command is exercised (n >= len(cmds) recommended)
        idx = [i % len(cmds) for i in range(n)]
        forced = torch.tensor([cmds[i] for i in idx], dtype=torch.float32,
                              device=device)
        env_key = [heldout_keys[i % len(heldout_keys)] for i in range(n)]

    # ── fixed deterministic rollout ────────────────────────────────
    tracked_hits = 0          # steps within tolerance (tracking_within_tol)
    tracked_total = 0
    err_sum = 0.0
    rew_sum = 0.0
    n_steps = 0
    # R2: per-condition (per forced command key) hit/total, so a single
    # dominating command can't hide inside the aggregate (SONIC lesson).
    per_hits: dict = {}
    per_total: dict = {}

    obs, _ = env.reset()
    if forced is not None and cmd_term is not None:
        _force_commands(cmd_term, forced)
    states = env.state()
    for _ in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = policy(obs, states)
            obs, rew, _, _, _ = env.step(actions)
            # re-force AFTER the step so the env's resample/zeroing is overridden
            if forced is not None and cmd_term is not None:
                _force_commands(cmd_term, forced)
            states = env.state()
        err = _tracked_command_error(env)
        if err is not None:
            within = (err < tol)
            tracked_hits += int(within.sum().item())
            tracked_total += err.numel()
            err_sum += float(err.mean().item())
            if forced is not None:   # bucket by forced command key
                w = within.tolist()
                for i, key in enumerate(env_key):
                    per_total[key] = per_total.get(key, 0) + 1
                    per_hits[key] = per_hits.get(key, 0) + int(w[i])
        rew_sum += float(torch.as_tensor(rew).float().mean().item())
        n_steps += 1

    metrics = {"n_steps": n_steps,
               "mean_reward": rew_sum / max(n_steps, 1)}
    if per_total:
        metrics["per_condition"] = {
            k: round(per_hits[k] / per_total[k], 6) for k in sorted(per_total)}
    if tracked_total > 0:
        rate = tracked_hits / tracked_total
        metrics["mean_tracking_error"] = err_sum / max(n_steps, 1)
    else:
        # non-locomotion task: fall back to mean reward as the gate signal
        rate = metrics["mean_reward"]

    if args_cli.heldout_manifest:
        # protected-metric pass: report under the heldout name + the keys
        # ACTUALLY forced this run (not the full manifest) so the watcher's
        # subset-integrity guard reflects real coverage (fix-verify low).
        metrics["heldout_success_rate"] = rate
        metrics["eval_keys"] = sorted(set(env_key))
        if set(env_key) != set(heldout_keys):
            metrics["heldout_coverage_partial"] = True
            print(f"[eval_rollout] WARNING: forced {len(set(env_key))} of "
                  f"{len(heldout_keys)} held-out commands (num_envs < grid); "
                  "raise --num_envs to cover the full grid")
    else:
        metrics[args_cli.gate_metric] = rate

    os.makedirs(args_cli.out_dir, exist_ok=True)
    out_path = os.path.join(args_cli.out_dir, "metrics_eval.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval_rollout] wrote {out_path}: "
          f"{ {k: v for k, v in metrics.items() if k != 'eval_keys'} }")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

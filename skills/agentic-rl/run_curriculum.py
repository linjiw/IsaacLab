# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Driver: assemble the run-manager loop over Isaac Lab and run one campaign.

Wires the vendored RunManager (core loop + guardrails) to the Isaac Lab
EngineAdapter, the knob registry, and one of three policy arms:

  control  - proposals disabled (fixed defaults) — the baseline
  manager  - a digest-reading band-stepper (the LLM's deterministic core)
  scripted - open-loop replay of the manager's own knob ladder (the arm the
             manager must BEAT to justify itself; see the SONIC null result)

Run on the host to build/inspect the loop with a mock adapter (no GPU), or in
the isaac-lab-base container against the real IsaacLabAdapter.

  # dry compose check (no GPU): mock adapter, 3 segments, manager arm
  python3 run_curriculum.py --dry-run --arm manager --segments 3

  # real short run (inside container, needs GPU):
  /isaac-sim/python.sh run_curriculum.py --arm control \
      --task Isaac-Velocity-Rough-Anymal-C-v0 --framework skrl \
      --segments 2 --iterations 30 --num-envs 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "runmanager"))
sys.path.insert(0, os.path.join(_HERE, "isaaclab-job-adapter"))
sys.path.insert(0, os.path.join(_HERE, "isaaclab-task-catalog"))

import yaml  # noqa: E402

from core.loop import LoopConfig, RunManager, control_config, scripted_config  # noqa: E402
from core.registry import KnobRegistry  # noqa: E402

REGISTRY = os.path.join(_HERE, "isaaclab-knob-registry", "registry.yaml")


def load_registry(path: str = REGISTRY) -> KnobRegistry:
    with open(path) as f:
        spec = yaml.safe_load(f)
    fams = (spec.get("meta") or {}).get("heldout_gated_families", ("schedule",))
    return KnobRegistry(spec, heldout_gated_families=tuple(fams))


def make_policy(arm: str, args):
    """The three arms, using the Isaac Lab-native policies. control needs no
    policy; scripted needs a ladder DERIVED from a real manager journal
    (review B4) — passed via --scripted-from-journal."""
    sys.path.insert(0, _HERE)
    from isaaclab_policies import (IsaacLabScriptedPolicy,
                                   LocomotionManagerPolicy, ladder_from_journal)
    if arm == "manager":
        # t_low/t_high are calibrated to the task's ACHIEVABLE held-out ceiling
        # (rough-terrain Anymal-C tracking-within-0.25-tol may plateau well
        # below 0.85), so expose them; defaults match the policy's 0.50/0.85.
        return LocomotionManagerPolicy(t_low=args.t_low, t_high=args.t_high,
                                       sustain=args.sustain)
    if arm == "scripted":
        if not args.scripted_from_journal:
            raise SystemExit(
                "the scripted arm requires --scripted-from-journal <manager "
                "journal.json>: its ladder must be DERIVED from a real "
                "manager run's realized decisions (review B4), never "
                "hand-authored. Run the manager arm first with --journal-out.")
        journal = json.load(open(args.scripted_from_journal))
        ladder = ladder_from_journal(journal)
        if not ladder:
            raise SystemExit(
                f"{args.scripted_from_journal} has no applied decisions to "
                "replay — the manager made no moves; nothing to ablate.")
        return IsaacLabScriptedPolicy(ladder=ladder)
    return None  # control: propose disabled


def make_config(arm: str, args, adapter) -> LoopConfig:
    common = dict(
        iterations_per_segment=args.iterations,
        eval_envs=args.eval_envs,
        run_eval=not args.no_eval,
        seed=args.seed,
        initial_checkpoint=args.warm_start,   # shared warm-start (review B4/G5)
        # Isaac Lab train scalar keys the digest should summarize
        train_scalar_keys=("Episode/rew_mean", "Episode/len_mean"),
    )
    # review R4: arm the disk gate (>=8GB free before each launch) on a real
    # run so a multi-segment campaign has the advertised backstop. The mock
    # adapter has no host dir, so only wire it for non-dry runs.
    if not args.dry_run:
        common["disk_gate_path"] = os.path.join("/workspace/isaaclab", "logs")
    # PROTECTED held-out metric (review B3). The hook runs the held-out eval
    # pass (forced frozen command grid) and returns the protected record; the
    # loop's __post_init__ REQUIRES an explicit standard-eval pin alongside it,
    # so eval_extra_overrides pins the training-mutable command range back to
    # stock for the STANDARD eval (defense-in-depth pin-the-scoreboard, G7).
    if args.heldout_manifest and not args.no_eval:
        common["heldout_hook"] = _make_heldout_hook(args)
        common["heldout_summary_keys"] = ("heldout_success_rate",
                                          "heldout_mean_tracking_error",
                                          "heldout_per_condition")
        common["eval_extra_overrides"] = [
            "env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0]"]
    if arm == "control":
        return control_config(**common)
    if arm == "scripted":
        return scripted_config(**common)
    return LoopConfig(arm="manager", **common)


def _make_heldout_hook(args):
    """Build the loop's heldout_hook: run the adapter's eval on the FROZEN
    held-out command grid (raw=True), then turn it into the protected record
    (integrity-guarded). Signature per loop.py: hook(adapter, seg, it, envs)."""
    sys.path.insert(0, os.path.join(_HERE, "isaaclab-heldout-watcher"))
    from holdout import heldout_record_from_metrics_eval, load_manifest
    manifest = load_manifest(args.heldout_manifest)

    def hook(adapter, seg, it, num_envs):
        raw = adapter.eval_segment(
            seg, it=it, num_envs=num_envs, out_suffix="_heldout", raw=True,
            extra_overrides=[f"--heldout_manifest={args.heldout_manifest}"])
        return heldout_record_from_metrics_eval(raw, manifest, it=it)

    return hook


def make_adapter(args):
    if args.dry_run:
        # scripted gate series that stays >= t_high so the manager arm HARDENS
        # (widens the command range) — exercises the full set/tripwire path.
        return MockLocomotionAdapter(gate_script=[0.9, 0.9, 0.9, 0.88, 0.9, 0.9],
                                     heldout=bool(args.heldout_manifest))
    from isaaclab_adapter import IsaacLabAdapter
    return IsaacLabAdapter(
        framework=args.framework, task=args.task, num_envs=args.num_envs,
        seed=args.seed, gate_metric=args.gate_metric)


class MockLocomotionAdapter:
    """CPU mock: Isaac Lab-flavored train records + a scripted eval gate
    series. Lets the whole loop (all three arms) run without Isaac Sim, using
    Isaac Lab knob/metric names (unlike the SONIC-specific vendored fakes)."""

    def __init__(self, gate_script, heldout=False, flat_reward=False):
        from core.protocols import ParsedSegment, Segment
        self._ParsedSegment = ParsedSegment
        self._Segment = Segment
        self.gate_script = list(gate_script)
        self.heldout = heldout
        # flat_reward: hold Episode/rew_mean constant so the digest reward
        # trend reads 'flat' — exercises the manager's lr-anneal branch.
        self.flat_reward = flat_reward
        self.launched = []
        self.i = 0
        self.j = 0

    def launch_segment(self, name, iterations, knobs, checkpoint_in=None):
        self.launched.append((name, dict(knobs), checkpoint_in))
        return self._Segment(name=name, iterations=iterations, knobs=dict(knobs),
                             checkpoint_in=checkpoint_in, status="running")

    def wait(self, seg, poll_s=0, timeout_s=0):
        seg.status = "done"
        seg.snapshot = f"/mock/{seg.name}/snapshot.pt"
        seg.experiment_dir = f"/mock/{seg.name}"
        return seg

    def parse_segment(self, seg):
        rew = 10.0 if self.flat_reward else 5.0 + 2.0 * self.i
        self.i += 1
        train = [{"it": k + 1, "Episode/rew_mean": rew, "Episode/len_mean": 200.0,
                  "Episode_Termination/base_contact": 0.1} for k in range(3)]
        return self._ParsedSegment(train=train, sampler=[])

    def eval_segment(self, seg, it, num_envs=64, extra_overrides=None,
                     out_suffix="_eval", raw=False):
        g = self.gate_script[min(self.j, len(self.gate_script) - 1)]
        self.j += 1
        # standard eval record; the held-out hook (raw=True) gets a held-out-
        # shaped dict so heldout_record_from_metrics_eval can build the record.
        if raw and self.heldout:
            # emit a per_condition dict with realistic spread so a dry-run
            # pilot fully rehearses verify_gate (incl. the non-degenerate gate)
            pc = {"vx=0.5,vy=0.0,wz=0.5": round(g + 0.05, 4),
                  "vx=1.0,vy=0.0,wz=1.0": round(max(g - 0.05, 0.0), 4)}
            return {"heldout_success_rate": g, "mean_tracking_error": 1.0 - g,
                    "per_condition": pc, "eval_keys": []}
        return {"it": it, "success_rate": g, "tracking": g, "mean_reward": 10.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Isaac Lab curriculum-RL driver")
    p.add_argument("--arm", choices=["control", "manager", "scripted"], default="manager")
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--iterations", type=int, default=50, help="train iters per segment")
    p.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-v0")
    p.add_argument("--framework", default="skrl", choices=["skrl", "rl_games"])
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--eval-envs", type=int, default=64)
    p.add_argument("--gate-metric", default="tracking")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="mock adapter, no GPU")
    p.add_argument("--journal-out", default=None)
    p.add_argument("--warm-start", default=None,
                   help="shared warm-start checkpoint for all arms (review B4/G5)")
    p.add_argument("--heldout-manifest", default=None,
                   help="held-out watcher manifest → protected heldout_success_rate")
    p.add_argument("--scripted-from-journal", default=None,
                   help="manager journal.json to derive the scripted ladder from")
    p.add_argument("--t-low", type=float, default=0.50,
                   help="manager EASE threshold (held-out success); calibrate "
                        "to the task's achievable ceiling")
    p.add_argument("--t-high", type=float, default=0.85,
                   help="manager HARDEN threshold (held-out success)")
    p.add_argument("--sustain", type=int, default=3,
                   help="consecutive evals the held-out condition must hold "
                        "before the manager acts (lower = acts sooner, fewer "
                        "segments needed for a non-vacuous run)")
    args = p.parse_args(argv)

    # Phase-1 is skrl-only. rl_games train/eval paths are not yet implemented
    # (eval_rollout raises; the train experiment-name override needs the '+'
    # append form) — fail LOUD here rather than launching a silent-null run
    # (review B6).
    if args.framework == "rl_games" and not args.dry_run:
        p.error("rl_games is not yet supported for a real Phase-1 run "
                "(eval + experiment-name override are unimplemented); use "
                "--framework skrl. See isaaclab-job-adapter/SKILL.md.")

    if args.arm in ("manager", "scripted") and not args.heldout_manifest and not args.dry_run:
        # BOTH the manager and its scripted competitor must use the identical
        # protected-metric gate, else the ON-vs-OFF comparison isn't apples-to-
        # apples (review B3 + fix-verify). The manager gates its decisions on
        # the held-out metric; the scripted arm's tripwire must guard the same
        # metric. Refuse a real run of either without it.
        p.error(f"the {args.arm} arm requires --heldout-manifest for a "
                "trustworthy run (both arms gate/guard on the protected "
                "held-out metric). Build one with holdout.py make-manifest.")

    # ARCHITECTURE preflight (real runs): the adapter shells into the
    # isaac-lab-base container via `docker exec`, so this driver must run ON
    # THE HOST (where `docker` exists), NOT inside the container. Fail loud
    # with a clear message rather than a raw FileNotFoundError deep in a
    # subprocess call.
    if not args.dry_run:
        import shutil as _sh
        if _sh.which("docker") is None:
            p.error("`docker` not found on PATH. The driver runs ON THE HOST "
                    "and execs into the isaac-lab-base container — do NOT run "
                    "it inside the container. Use the host Python "
                    "(/tmp/rmc-venv/bin/python), not /isaac-sim/python.sh.")

    registry = load_registry()
    adapter = make_adapter(args)
    policy = make_policy(args.arm, args)
    config = make_config(args.arm, args, adapter)

    mgr = RunManager(policy=policy, adapter=adapter, registry=registry,
                     config=config)
    summary = mgr.run(args.segments)

    # headline = the PROTECTED held-out series (review B3), not the eval signal
    heldout_series = [(e.get("heldout") or {}).get("heldout_success_rate")
                      for e in mgr.journal]
    summary["heldout_success_series"] = heldout_series
    summary["headline_metric"] = ("heldout_success_rate"
                                  if args.heldout_manifest else "NONE (untrustworthy)")

    print(json.dumps(summary, indent=2, default=str))
    if args.journal_out:
        from core.journal import save_journal
        save_journal(mgr.journal, args.journal_out)
        print(f"[run_curriculum] journal → {args.journal_out} "
              f"({len(mgr.journal)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

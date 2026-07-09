# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Isaac Lab EngineAdapter: launch / observe / eval / rollback lifecycle for
LLM-guided curriculum-RL run-segments.

Implements runmanager/core/protocols.py::EngineAdapter for Isaac Lab's
rl_games and skrl training scripts. Structure mirrors the verified
sonic-job-adapter: Isaac Lab here runs INSIDE the `isaac-lab-base` docker
container (Isaac Sim kit python at /isaac-sim/python.sh, Isaac Lab at
/workspace/isaaclab), so launch/observe use docker-exec exactly as SONIC did.

Three responsibilities, split by testability:

1. **Command building (pure, CPU-tested)** — build the train.py / eval
   invocation with knob values mapped from registry names to Hydra override
   paths (KNOB_TO_HYDRA, per framework). Never invents a config path.
2. **Scalar normalization (pure, CPU-tested)** — map Isaac Lab TensorBoard
   tags to digest train-stream keys.
3. **Docker plumbing (thin, integration-only)** — launch subprocess, poll,
   read logs / TB events / resolved config, snapshot checkpoints. Requires
   the running container + a GPU; not unit-tested.

Framework deltas (verified against scripts/reinforcement_learning/*/train.py):

  rl_games: logs/rl_games/<config.name>/<ts>/, checkpoints nn/*.pth,
            knob prefix agent.params.config.*, seed agent.params.seed,
            --max_iterations -> params.config.max_epochs, resume --checkpoint.
  skrl:     logs/skrl/<directory>/<ts>_<algo>_<framework>/, checkpoints
            checkpoints/*.pt, knob prefix agent.*, seed agent.seed,
            --max_iterations handled by the script (timesteps = it*rollouts),
            resume --checkpoint. --algorithm/--ml_framework select the agent.

Both dump params/{env,agent}.yaml (resolved config -> drift verification).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

# canonical TB-scalar normalization lives in the run-digest skill (single
# source of truth); import it so the adapter and that skill never drift.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "isaaclab-run-digest"))
from tb_reader import normalize_tag, records_from_scalars  # noqa: E402,F401

# ── container / path constants (this box's isaac-lab-base, verified) ──
CONTAINER = "isaac-lab-base"
ISAACLAB_DIR = "/workspace/isaaclab"          # Isaac Lab root INSIDE container
PYTHON_SH = "/isaac-sim/python.sh"            # Isaac Sim kit python
# the agentic-rl skill dir as seen inside the container (host /workspace is
# bind-mounted at the same path); used to reach the bundled eval script
SKILL_DIR = "/workspace/IsaacLab/skills/agentic-rl/isaaclab-job-adapter"
EVAL_SCRIPT = f"{SKILL_DIR}/eval_rollout.py"

SUPPORTED_FRAMEWORKS = ("skrl", "rl_games")


# ── knob → Hydra override path, per framework ────────────────────────
# env.* knobs are framework-independent (the env cfg is shared); agent.*
# knobs differ (rl_games nests under params.config, skrl under agent).
# Extend these tables as the registry grows; build_overrides RAISES on any
# knob without a mapping — the adapter never guesses a config path.
#
# IMPORTANT: because a decision is applied by RELAUNCHING the next segment
# with a Hydra override (not by mutating a live process), a knob does NOT
# require a native CurriculumTermCfg to take effect — any config leaf works.
# Paths below are VERIFIED against the locomotion-velocity Anymal-C configs
# (velocity_env_cfg.py + config/anymal_c/agents/{skrl,rl_games}_rough_*.yaml).
_ENV_KNOBS: Dict[str, str] = {
    # Family A — data/task curriculum (command ranges are tuples → "[lo, hi]")
    "command_range_lin_vel_x": "env.commands.base_velocity.ranges.lin_vel_x",
    "command_range_lin_vel_y": "env.commands.base_velocity.ranges.lin_vel_y",
    "command_range_ang_vel_z": "env.commands.base_velocity.ranges.ang_vel_z",
    # Family B — reward-weight ramps (env.rewards.<term>.weight; verified terms)
    "reward_weight_action_rate": "env.rewards.action_rate_l2.weight",
    "reward_weight_dof_acc": "env.rewards.dof_acc_l2.weight",
    "reward_weight_flat_orientation": "env.rewards.flat_orientation_l2.weight",
}
_AGENT_KNOBS: Dict[str, Dict[str, str]] = {
    # Family C — optimizer meta-params (framework-specific prefix)
    "skrl": {
        "learning_rate": "agent.agent.learning_rate",
        "entropy_coef": "agent.agent.entropy_loss_scale",
        # KLAdaptiveLR target lives under the scheduler kwargs, not top-level
        "desired_kl": "agent.agent.learning_rate_scheduler_kwargs.kl_threshold",
    },
    "rl_games": {
        "learning_rate": "agent.params.config.learning_rate",
        "entropy_coef": "agent.params.config.entropy_coef",
        "desired_kl": "agent.params.config.kl_threshold",
    },
}

# Command-range knobs are modeled in the registry as a scalar HALF-WIDTH
# (the validator supports only float|choice, not tuples). At launch the
# scalar w expands to the symmetric range "[-w, w]"; and because the
# resolved config leaf is a tuple, these knobs are EXCLUDED from exact-match
# config-drift verification (a scalar belief can never equal a tuple).
_RANGE_KNOBS = frozenset({
    "command_range_lin_vel_x", "command_range_lin_vel_y",
    "command_range_ang_vel_z",
})


def knob_to_hydra(framework: str) -> Dict[str, str]:
    """The full registry-knob -> Hydra-override-path table for `framework`."""
    _check_framework(framework)
    out = dict(_ENV_KNOBS)
    out.update(_AGENT_KNOBS[framework])
    return out


def knob_to_config_path(framework: str) -> Dict[str, str]:
    """Same knobs as dotted paths into the resolved params/{env,agent}.yaml.

    Isaac Lab dumps env cfg and agent cfg to SEPARATE files, so the leading
    'env.'/'agent.' segment is stripped here — the caller verifies each knob
    against whichever resolved file its path targets. Hydra append markers
    ('+'/'++') are not used by these knobs (all target existing leaves).

    Range knobs are OMITTED: their scalar half-width belief cannot exact-match
    the resolved tuple leaf, so including them would raise a spurious
    ConfigDriftError every tick."""
    return {name: path.split(".", 1)[1] if "." in path else path
            for name, path in knob_to_hydra(framework).items()
            if name not in _RANGE_KNOBS}


def _check_framework(framework: str) -> None:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"unsupported framework {framework!r}; expected one of "
            f"{SUPPORTED_FRAMEWORKS}")


def build_overrides(knobs: Dict[str, Any], framework: str) -> List[str]:
    """Map registry knob values to Hydra `key=value` override strings.

    Raises KeyError on any knob without a mapping for this framework — the
    adapter must never invent config paths (mirrors sonic-job-adapter)."""
    table = knob_to_hydra(framework)
    out: List[str] = []
    for name, value in sorted(knobs.items()):
        if name not in table:
            raise KeyError(
                f"knob {name!r} has no Hydra mapping for framework "
                f"{framework!r}; verify the config path against a resolved "
                "params/*.yaml before adding it to the adapter tables")
        # range knobs: a scalar half-width w -> symmetric range [-w, w]
        if name in _RANGE_KNOBS and isinstance(value, (int, float)):
            w = abs(float(value))
            value = (-w, w)
        out.append(f"{table[name]}={_fmt(value)}")
    return out


def _fmt(value: Any) -> str:
    """Render a knob value as a Hydra CLI token. Lists/tuples become the
    SPACE-FREE list form ("[a,b]"): Hydra parses it, and it survives bash
    word-splitting inside `docker exec bash -c` (a space would split the
    token and abort the launch — see build_train_command quoting)."""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


# ── train / eval command building (pure) ─────────────────────────────
def build_train_command(
    framework: str,
    task: str,
    experiment_name: str,
    iterations: int,
    knobs: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    num_envs: Optional[int] = None,
    seed: int = 42,
    algorithm: str = "PPO",
    ml_framework: str = "torch",
    log_path: Optional[str] = None,
    extra_overrides: Optional[Sequence[str]] = None,
) -> List[str]:
    """The docker-exec training launch for `framework`.

    `seed` is pinned explicitly: arm-comparison experiments depend on
    identical-prefix segments, which must not rest on an unpinned default.
    `extra_overrides` are appended verbatim AFTER the knob overrides (the
    held-out wiring uses them to pin the training task set); they are NOT in
    the manager's action space."""
    _check_framework(framework)
    script = f"scripts/reinforcement_learning/{framework}/train.py"
    flags: List[str] = [f"--task={shlex.quote(task)}", "--headless",
                        f"--max_iterations={int(iterations)}",
                        f"--seed={int(seed)}"]
    if num_envs is not None:
        flags.append(f"--num_envs={int(num_envs)}")
    if checkpoint:
        flags.append(f"--checkpoint={shlex.quote(checkpoint)}")
    if framework == "skrl":
        flags += [f"--algorithm={algorithm}", f"--ml_framework={ml_framework}"]
    # run name so the log dir is discoverable; rl_games reads
    # agent.params.config.full_experiment_name, skrl reads the experiment name
    exp_override = (f"agent.params.config.full_experiment_name={experiment_name}"
                    if framework == "rl_games"
                    else f"agent.agent.experiment.experiment_name={experiment_name}")
    overrides = [exp_override]
    if framework == "skrl":
        # R1: write_interval='auto' resolves to 0 for short segments, dropping
        # the entire TB train stream (so rew_mean_last=None and tripwire
        # baselines can't arm). Pin it to 1 so every short segment logs.
        overrides.append("agent.agent.experiment.write_interval=1")
    overrides += build_overrides(knobs or {}, framework)
    overrides += list(extra_overrides or [])
    inner_parts = [f"cd {ISAACLAB_DIR} &&"]
    if log_path:
        inner_parts.append("nohup")
    # shlex.quote every override token: a Hydra list value like "[-1.5,1.5]"
    # (or any override carrying shell metacharacters) must reach the python
    # process as ONE argument, not be word-split by `docker exec bash -c`.
    inner_parts += [PYTHON_SH, script, *flags,
                    *[shlex.quote(o) for o in overrides]]
    if log_path:
        inner_parts.append(f"> {shlex.quote(log_path)} 2>&1 &")
    inner = " ".join(inner_parts)
    return ["docker", "exec", CONTAINER, "bash", "-c", inner]


def build_eval_command(
    framework: str,
    task: str,
    checkpoint: str,
    out_dir: str,
    num_envs: int = 64,
    eval_steps: int = 1000,
    gate_metric: Optional[str] = None,
    seed: int = 1234,
    algorithm: str = "PPO",
    ml_framework: str = "torch",
    log_path: Optional[str] = None,
    extra_overrides: Optional[Sequence[str]] = None,
) -> List[str]:
    """The eval-only headless rollout invocation.

    Isaac Lab's stock play.py loops forever and writes NO metrics file, so we
    invoke the bundled eval_rollout.py, which loads a checkpoint, runs a
    fixed deterministic rollout, and writes <out_dir>/metrics_eval.json.

    Pin-the-scoreboard (the SONIC M1 lesson): eval must run at fixed settings
    the manager cannot touch. Use the task's `-Play-v0` variant (reduced
    envs, noise/DR disabled) and pass NO manager knob overrides here; any
    termination/curriculum term the manager can mutate must be re-pinned via
    `extra_overrides` at its stock value so it can't leak into the score."""
    _check_framework(framework)
    flags = [
        f"--framework={framework}", f"--task={shlex.quote(task)}",
        f"--checkpoint={shlex.quote(checkpoint)}",
        f"--out_dir={shlex.quote(out_dir)}",
        f"--num_envs={int(num_envs)}", f"--eval_steps={int(eval_steps)}",
        f"--seed={int(seed)}", "--headless",
    ]
    if gate_metric:
        flags.append(f"--gate_metric={shlex.quote(gate_metric)}")
    if framework == "skrl":
        flags += [f"--algorithm={algorithm}", f"--ml_framework={ml_framework}"]
    inner_parts = [f"cd {ISAACLAB_DIR} &&"]
    if log_path:
        inner_parts.append("nohup")
    # quote eval pins too (a range-valued re-pin has the same word-split risk)
    inner_parts += [PYTHON_SH, EVAL_SCRIPT, *flags,
                    *[shlex.quote(o) for o in (extra_overrides or [])]]
    if log_path:
        inner_parts.append(f"> {shlex.quote(log_path)} 2>&1 &")
    inner = " ".join(inner_parts)
    return ["docker", "exec", CONTAINER, "bash", "-c", inner]


# ── scalar normalization (pure) ──────────────────────────────────────
# normalize_tag + records_from_scalars are imported at the top from the
# run-digest skill's tb_reader (single source of truth) — kept there so the
# adapter and that skill can never drift on Isaac Lab's TB tag names.


# ── log / traceback parsing (pure) ───────────────────────────────────
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_EXP_LINE = re.compile(r"Exact experiment name requested from command line:\s*(.+)")
_LOGDIR_LINE = re.compile(r"Logging experiment in directory:\s*(.+)")


@dataclasses.dataclass
class ParsedConsole:
    experiment_name: Optional[str] = None
    log_root: Optional[str] = None
    tracebacks: int = 0


def parse_console_log(lines: Iterable[str]) -> ParsedConsole:
    """Extract run health + log-dir discovery from stdout/stderr.

    Both frameworks print 'Logging experiment in directory: <root>' and
    'Exact experiment name requested from command line: <name>' — together
    they locate the run dir. A Traceback marks the segment failed."""
    out = ParsedConsole()
    for raw in lines:
        line = raw.rstrip("\n")
        if _TRACEBACK.search(line):
            out.tracebacks += 1
        m = _EXP_LINE.search(line)
        if m:
            out.experiment_name = m.group(1).strip()
        m = _LOGDIR_LINE.search(line)
        if m:
            out.log_root = m.group(1).strip()
    return out


def parse_metrics_eval(metrics: Dict[str, Any], it: int,
                       gate_metric: str = "gate") -> Dict[str, Any]:
    """One eval_rollout metrics_eval.json -> one digest eval-stream record.

    The eval script writes a flat dict of scalar metrics plus optional
    per-key breakdowns. `gate_metric` names which scalar becomes the loop's
    `success_rate` (the tripwire/promotion signal); it is copied under both
    its own name and 'success_rate' so the digest's eval section populates."""
    rec: Dict[str, Any] = {"it": it}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            rec[k] = float(v)
    if gate_metric in metrics and isinstance(metrics[gate_metric], (int, float)):
        rec["success_rate"] = float(metrics[gate_metric])
    if isinstance(metrics.get("failed_keys"), list):
        rec["failed_keys"] = list(metrics["failed_keys"])
    # R2: pass the per-condition breakdown through (a dict, previously dropped)
    # so the loop can surface per-terrain / per-command-bin deltas and detect a
    # single dominating condition — the artifact that killed the SONIC claim.
    if isinstance(metrics.get("per_condition"), dict):
        rec["per_condition"] = metrics["per_condition"]
    return rec


# Isaac Lab's dump_yaml emits !!python/* tags that yaml.safe_load refuses.
# We only need scalar knob VALUES from the resolved config for drift
# verification, so parse with a loader that accepts every !!python/* tag by
# constructing a plain, value-faithful Python object:
#   - !!python/tuple            -> list (range values preserved; value-equal)
#   - !!python/object/apply:... -> the applied node's args (or None) — a
#     callable/object we never compare a knob value against, so lossy is fine
#   - !!python/name:... , !!python/object:... -> None
# A dedicated loader (not regex) handles YAML STRUCTURE correctly, unlike a
# line strip (which broke on block mappings under a tagged key).
class _SafeIsaacLoader(yaml.SafeLoader):
    pass


def _construct_tuple(loader, node):
    return loader.construct_sequence(node)


def _construct_drop(loader, tag_suffix, node):
    # multi-constructor: signature is (loader, tag_suffix, node). object/name/
    # apply nodes -> a best-effort plain value, else None (no code execution).
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return None


_SafeIsaacLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_tuple)
# every other python/* tag -> dropped to a plain value
_SafeIsaacLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_drop)


def _safe_load_isaac_yaml(text: str) -> Any:
    """yaml.load of an Isaac Lab dump_yaml doc, tolerating its !!python/* tags
    (tuple->list, everything else->plain value). Safe: no code execution — the
    loader constructs plain containers/None, never calls the tagged callables."""
    return yaml.load(text, Loader=_SafeIsaacLoader)


def write_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── segment dataclass (matches core.protocols.Segment field set) ─────
@dataclasses.dataclass
class Segment:
    name: str
    iterations: int
    knobs: Dict[str, Any]
    checkpoint_in: Optional[str] = None
    log_path: str = ""
    experiment_dir: Optional[str] = None
    snapshot: Optional[str] = None
    status: str = "pending"


@dataclasses.dataclass
class ParsedSegment:
    train: List[dict] = dataclasses.field(default_factory=list)
    sampler: List[dict] = dataclasses.field(default_factory=list)
    checkpoint_loaded_step: Optional[int] = None
    experiment_dir: Optional[str] = None
    tracebacks: int = 0


# ── docker plumbing (thin, integration-only — needs the container) ───
def _dexec(cmd: str, timeout: int = 120) -> str:
    proc = subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", cmd],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker exec failed ({proc.returncode}): {proc.stderr[:300]}")
    return proc.stdout


def _checkpoint_glob(framework: str, run_dir: str) -> str:
    """Glob for the numbered checkpoints of a run. skrl writes
    checkpoints/agent_<step>.pt (+ best_agent.pt, which we exclude so resume
    picks the LATEST by step, not the best-by-return); rl_games writes
    nn/<name>.pth."""
    if framework == "rl_games":
        return f"{run_dir}/nn/*.pth"
    return f"{run_dir}/checkpoints/agent_*.pt"


class IsaacLabAdapter:
    """EngineAdapter for Isaac Lab rl_games/skrl via docker-exec.

    Command building + parsing (the tested surface) live in the module-level
    pure functions; this class wires them to the container. launch/wait/eval
    require the running isaac-lab-base container and a GPU."""

    def __init__(self, framework: str, task: str,
                 num_envs: Optional[int] = None, seed: int = 42,
                 algorithm: str = "PPO", ml_framework: str = "torch",
                 gate_metric: str = "gate", eval_task: Optional[str] = None,
                 eval_extra_overrides: Optional[Sequence[str]] = None,
                 extra_overrides: Optional[Sequence[str]] = None):
        _check_framework(framework)
        self.framework = framework
        self.task = task
        self.eval_task = eval_task or task.replace("-v0", "-Play-v0")
        self.num_envs = num_envs
        self.seed = seed
        self.algorithm = algorithm
        self.ml_framework = ml_framework
        self.gate_metric = gate_metric
        self.eval_extra_overrides = list(eval_extra_overrides or [])
        self.extra_overrides = list(extra_overrides or [])
        self.log_root = os.path.join(
            ISAACLAB_DIR, "logs", framework)
        self.segments: List[Segment] = []

    # -- EngineAdapter surface --------------------------------------
    def knob_to_config_path(self) -> Dict[str, str]:
        return knob_to_config_path(self.framework)

    def launch_segment(self, name: str, iterations: int,
                       knobs: Dict[str, Any],
                       checkpoint_in: Optional[str] = None) -> Segment:
        log_path = f"{ISAACLAB_DIR}/logs/_seg_{name}.log"
        seg = Segment(name=name, iterations=iterations, knobs=dict(knobs),
                      checkpoint_in=checkpoint_in, log_path=log_path)
        cmd = build_train_command(
            self.framework, self.task, experiment_name=name,
            iterations=iterations, knobs=knobs, checkpoint=checkpoint_in,
            num_envs=self.num_envs, seed=self.seed, algorithm=self.algorithm,
            ml_framework=self.ml_framework, log_path=log_path,
            extra_overrides=self.extra_overrides)
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=120)
        seg.status = "running"
        self.segments.append(seg)
        return seg

    def wait(self, seg: Segment, poll_s: int = 20,
             timeout_s: int = 3600) -> Segment:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if not self._is_training_running():
                break
            time.sleep(poll_s)
        else:
            seg.status = "failed"
            raise TimeoutError(
                f"segment {seg.name} still running after {timeout_s}s")
        parsed = self.parse_segment(seg)
        seg.experiment_dir = parsed.experiment_dir
        # FAIL LOUD (review B2): a crashed launch may leave no 'Traceback'
        # (e.g. a Hydra config-parse abort), so status must NOT rest on the
        # traceback count alone. A segment is 'done' only if it (a) had no
        # traceback, (b) produced a discoverable run dir for THIS segment, and
        # (c) that dir yields a fresh checkpoint. Otherwise the launch failed
        # and we must NOT silently reuse a prior segment's checkpoint.
        if parsed.tracebacks or not seg.experiment_dir:
            seg.status = "failed"
            return seg
        snap = self._snapshot_checkpoint(seg.experiment_dir, seg.name)
        if not snap:
            seg.status = "failed"
            return seg
        seg.snapshot = snap
        seg.status = "done"
        return seg

    def parse_segment(self, seg: Segment) -> ParsedSegment:
        lines = self._read_log(seg.log_path)
        console = parse_console_log(lines)
        run_dir = self._resolve_run_dir(console, seg_name=seg.name)
        train = self._read_train_records(run_dir) if run_dir else []
        return ParsedSegment(train=train, experiment_dir=run_dir,
                             tracebacks=console.tracebacks)

    def eval_segment(self, seg: Segment, it: int, num_envs: int = 64,
                     extra_overrides: Optional[List[str]] = None,
                     out_suffix: str = "_eval", raw: bool = False,
                     poll_s: int = 10, timeout_s: int = 1200) -> Dict[str, Any]:
        ckpt = seg.snapshot or (seg.experiment_dir
                                and self._latest_checkpoint(seg.experiment_dir))
        if not ckpt:
            raise RuntimeError(
                f"segment {seg.name} has no checkpoint to evaluate")
        out_dir = f"{ISAACLAB_DIR}/logs/_eval_{seg.name}{out_suffix}"
        log_path = f"{out_dir}.log"
        _dexec(f"rm -rf {shlex.quote(out_dir)} && mkdir -p {shlex.quote(out_dir)}")
        pins = list(self.eval_extra_overrides) + list(extra_overrides or [])
        cmd = build_eval_command(
            self.framework, self.eval_task, ckpt, out_dir, num_envs=num_envs,
            gate_metric=self.gate_metric, algorithm=self.algorithm,
            ml_framework=self.ml_framework, log_path=log_path,
            extra_overrides=pins)
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=120)
        time.sleep(min(poll_s, 10))
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if not self._is_eval_running():
                break
            time.sleep(poll_s)
        else:
            raise TimeoutError(
                f"eval for segment {seg.name} still running after {timeout_s}s")
        metrics = self._read_metrics_eval(out_dir, log_path)
        if raw:
            return metrics
        return parse_metrics_eval(metrics, it=it, gate_metric=self.gate_metric)

    def resolved_config_text(self, seg: Segment) -> Optional[str]:
        """Resolved config for the segment, as ONE yaml.safe_load-able doc.

        Isaac Lab dumps env.yaml + agent.yaml with dump_yaml, which emits
        Python-specific tags (`!!python/tuple`, `!!python/object/...`,
        `!!python/name:...`) that yaml.safe_load (used by the loop's config-
        drift verifier) REFUSES. We sanitize those tags to safe-loadable YAML.
        env.* knob paths resolve under `env:`, agent.* under `agent:`, so we
        NEST the two files under those keys — matching knob_to_config_path,
        which strips the leading env./agent. segment (i.e. it walks the path
        WITHOUT the prefix, so we must NOT re-add it). See knob_to_config_path:
        the stripped paths are relative to each file's own root, so we return
        the two docs merged at the top level via safe concatenation is WRONG
        (duplicate keys). Instead we load each independently and only expose
        the knobs' resolved values by re-serializing a merged safe mapping."""
        if not seg.experiment_dir:
            return None
        env_txt = self._cat(f"{seg.experiment_dir}/params/env.yaml")
        agent_txt = self._cat(f"{seg.experiment_dir}/params/agent.yaml")
        docs = []
        for txt in (env_txt, agent_txt):
            if not txt:
                continue
            try:
                d = _safe_load_isaac_yaml(txt)
            except yaml.YAMLError:
                continue
            if isinstance(d, dict):
                docs.append(d)
        if not docs:
            return None
        # merge env + agent at the top level; knob_to_config_path strips the
        # env./agent. prefix, so its paths are relative to each file's root and
        # the two roots don't collide (env cfg keys vs agent cfg keys differ).
        merged: Dict[str, Any] = {}
        for d in docs:
            merged.update(d)
        return yaml.safe_dump(merged, default_flow_style=False)

    # -- thin helpers (docker exec) ---------------------------------
    def _read_log(self, log_path: str) -> List[str]:
        return _dexec(f"cat {shlex.quote(log_path)} 2>/dev/null || true",
                      timeout=120).splitlines()

    def _cat(self, path: str) -> Optional[str]:
        out = _dexec(f"cat {shlex.quote(path)} 2>/dev/null || true")
        return out or None

    def _resolve_run_dir(self, console: ParsedConsole,
                         seg_name: Optional[str] = None) -> Optional[str]:
        """Locate a segment's run dir by its UNIQUE experiment-name suffix.
        Both frameworks APPEND the experiment name to a timestamped dir
        (skrl: <ts>_<algo>_<framework>_<name>; rl_games: <ts>_<name>), so we
        glob <log_root>/*_<name>.

        NO newest-dir fallback (review B2): if the run dir can't be found by
        this segment's own name, the launch almost certainly failed, and
        falling back to the newest dir would silently reuse a PRIOR segment's
        checkpoint. Returning None makes wait() mark the segment failed."""
        if not console.log_root or not seg_name:
            return None
        hit = _dexec(
            f"ls -dt {shlex.quote(console.log_root)}/*_{shlex.quote(seg_name)} "
            "2>/dev/null | head -1 || true").strip()
        return hit.rstrip("/") or None

    def _read_train_records(self, run_dir: str) -> List[dict]:
        """Read TB scalars from `run_dir` via the container's tbparse/
        tensorboard, emit normalized digest records. Uses a tiny in-container
        python snippet so the host needs no TB dependency."""
        script = (
            "import json,glob,sys;"
            "from tensorboard.backend.event_processing.event_accumulator "
            "import EventAccumulator;"
            f"files=glob.glob({run_dir!r}+'/**/events.out.tfevents.*',recursive=True);"
            "ea=EventAccumulator(sorted(files)[-1]) if files else None;"
            "d={};"
            "ea and ea.Reload();"
            "ea and [d.__setitem__(t,[(s.step,s.value) for s in ea.Scalars(t)]) "
            "for t in ea.Tags().get('scalars',[])];"
            "print(json.dumps(d))"
        )
        out = _dexec(f"{PYTHON_SH} -c {shlex.quote(script)} 2>/dev/null || echo '{{}}'",
                     timeout=180)
        try:
            scalars = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except (json.JSONDecodeError, IndexError):
            scalars = {}
        return records_from_scalars(scalars)

    def _latest_checkpoint(self, run_dir: str) -> Optional[str]:
        out = _dexec(
            f"ls -t {_checkpoint_glob(self.framework, run_dir)} 2>/dev/null | head -1 || true"
        ).strip()
        return out or None

    def _snapshot_checkpoint(self, run_dir: str, tag: str) -> Optional[str]:
        src = self._latest_checkpoint(run_dir)
        if not src:
            return None
        dst = f"{run_dir}/snapshot_{tag}{os.path.splitext(src)[1]}"
        _dexec(f"cp {shlex.quote(src)} {shlex.quote(dst)}", timeout=300)
        return dst

    def _read_metrics_eval(self, out_dir: str, log_path: str) -> Dict[str, Any]:
        try:
            out = _dexec(f"cat {shlex.quote(out_dir)}/metrics_eval.json",
                         timeout=60)
        except RuntimeError as e:
            raise RuntimeError(
                f"eval produced no metrics_eval.json (see {log_path}): {e}"
            ) from e
        return json.loads(out)

    def _is_training_running(self) -> bool:
        return self._pgrep("train.py")

    def _is_eval_running(self) -> bool:
        return self._pgrep("eval_rollout.py")

    def _pgrep(self, needle: str) -> bool:
        pat = f"[{needle[0]}]{needle[1:]}"  # bracket trick: skip the pgrep itself
        out = _dexec(f"pgrep -f {shlex.quote(pat)} | wc -l || true").strip()
        try:
            return int(out) > 0
        except ValueError:
            return False


# ── CLI (dry-run command building; no container needed) ──────────────
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Isaac Lab job adapter")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("command", help="print the train launch (dry run)")
    c.add_argument("--framework", default="skrl", choices=SUPPORTED_FRAMEWORKS)
    c.add_argument("--task", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--iterations", type=int, default=50)
    c.add_argument("--num-envs", type=int, default=None)
    c.add_argument("--checkpoint", default=None)
    c.add_argument("--knob", action="append", default=[], help="name=value, repeatable")

    e = sub.add_parser("eval-command", help="print the eval launch (dry run)")
    e.add_argument("--framework", default="skrl", choices=SUPPORTED_FRAMEWORKS)
    e.add_argument("--task", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out-dir", default="/tmp/eval")

    m = sub.add_parser("knobs", help="print the knob->hydra map for a framework")
    m.add_argument("--framework", default="skrl", choices=SUPPORTED_FRAMEWORKS)

    args = p.parse_args(argv)
    if args.cmd == "command":
        knobs = dict(kv.split("=", 1) for kv in args.knob)
        cmd = build_train_command(
            args.framework, args.task, experiment_name=args.name,
            iterations=args.iterations, knobs=knobs, checkpoint=args.checkpoint,
            num_envs=args.num_envs)
        print(" ".join(shlex.quote(c) for c in cmd))
    elif args.cmd == "eval-command":
        cmd = build_eval_command(args.framework, args.task, args.checkpoint,
                                 args.out_dir)
        print(" ".join(shlex.quote(c) for c in cmd))
    elif args.cmd == "knobs":
        print(json.dumps(knob_to_hydra(args.framework), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

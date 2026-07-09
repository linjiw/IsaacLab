# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Held-out eval watcher: the producer of the PROTECTED heldout_success_rate.

DESIGN §6 axiom "protect the manager's metric from the manager". Adapted from
the SONIC watcher (sonic-heldout-watcher/holdout.py) for Isaac Lab locomotion:
where SONIC held out discrete motion CLIPS, we hold out a frozen grid of
VELOCITY COMMANDS (lin_vel_x, lin_vel_y, ang_vel_z tuples). The held-out eval
FORCES those exact commands (the command term's vel_command_b is directly
settable), so the manager's training command-RANGE knob can never influence,
reweight, or filter the protected metric — protection is structural.

Three responsibilities (CPU-testable core; live wiring in SKILL.md +
eval_rollout.py --heldout):

1. `command_grid(...)` + `select_holdout(keys, fraction, salt)` — build the
   candidate command grid, then a deterministic salted hash split into
   held-out vs curriculum commands. Hash-based (stable under grid changes) and
   salted (composition not reproducible without the salt, which lives in the
   watcher's manifest — a file the manager never reads).
2. `write_manifest / load_manifest` — the watcher's private record: salt,
   fraction, held-out command list, integrity digest (load fails on tamper).
3. `heldout_record_from_metrics_eval(...)` — turn one held-out eval pass's
   metrics_eval.json into the single record the digest consumes
   ({"it","heldout_success_rate",...}). Refuses to emit a number if the eval
   demonstrably did not run on the held-out grid (foreign keys), so a
   mis-wired eval can never feed the manager a wrong protected metric.

A command key is the canonical string "vx=<>,vy=<>,wz=<>" (rounded), so keys
are hashable/serializable and map 1:1 to a forced command tuple.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MANIFEST_VERSION = "0.1.0"


# ── command-grid keys ────────────────────────────────────────────────
def command_key(vx: float, vy: float, wz: float, ndigits: int = 3) -> str:
    """Canonical string key for a velocity command (rounded for stability)."""
    return f"vx={round(vx, ndigits)},vy={round(vy, ndigits)},wz={round(wz, ndigits)}"


def parse_command_key(key: str) -> Tuple[float, float, float]:
    """Inverse of command_key: 'vx=..,vy=..,wz=..' -> (vx, vy, wz)."""
    parts = dict(p.split("=", 1) for p in key.split(","))
    return float(parts["vx"]), float(parts["vy"]), float(parts["wz"])


def command_grid(
    lin_vel_x: Sequence[float],
    ang_vel_z: Sequence[float],
    lin_vel_y: Sequence[float] = (0.0,),
) -> List[str]:
    """Cartesian product of the given axis samples → sorted command keys.

    Default holds lin_vel_y at 0 (Anymal-C velocity tasks are dominated by
    forward + yaw). The grid is the CANDIDATE pool; select_holdout partitions
    it.

    IMPORTANT (verified in-container 2026-07-09): choose grid points WITHIN the
    ACHIEVABLE tracking envelope. Protection comes from the manager not
    seeing/reweighting/re-thresholding the held-out commands — NOT from making
    them physically unreachable. A grid of e.g. vx=±3 m/s floors
    heldout_success_rate at 0 for every policy (tracking error ~1.9 >> tol
    0.25), so the metric cannot discriminate quality and the manager gates on a
    dead signal. In-envelope commands (|vx|<=1, |wz|<=1) give a non-degenerate,
    per-condition-varying metric."""
    seen, keys = set(), []
    for vx in lin_vel_x:
        for vy in lin_vel_y:
            for wz in ang_vel_z:
                k = command_key(vx, vy, wz)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return sorted(keys)


# ── salted hash split (mirrors SONIC) ────────────────────────────────
def _key_bucket(key: str, salt: str) -> float:
    """Map a key to a stable pseudo-uniform value in [0, 1)."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def select_holdout(keys: Iterable[str], fraction: float, salt: str) -> Dict[str, List[str]]:
    """Deterministic hash split. Same (key, salt) → same side, always."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if not salt:
        raise ValueError("salt must be non-empty (it is the composition secret)")
    heldout, curriculum = [], []
    seen = set()
    for k in keys:
        if k in seen:
            raise ValueError(f"duplicate command key: {k!r}")
        seen.add(k)
        (heldout if _key_bucket(k, salt) < fraction else curriculum).append(k)
    if not heldout or not curriculum:
        raise ValueError(
            f"degenerate split: {len(heldout)} held-out / {len(curriculum)} curriculum")
    return {"heldout": sorted(heldout), "curriculum": sorted(curriculum)}


def _digest_of(keys: List[str], salt: str) -> str:
    return hashlib.sha256((salt + "\n" + "\n".join(sorted(keys))).encode()).hexdigest()


def write_manifest(
    path: str,
    keys: Iterable[str],
    fraction: float,
    salt: str,
    tolerance: float = 0.25,
) -> Dict[str, Any]:
    """Create and persist the watcher's private manifest. Store OUTSIDE any
    directory the manager process reads (protection by boundary, not rule)."""
    split = select_holdout(keys, fraction, salt)
    manifest = {
        "version": MANIFEST_VERSION,
        "fraction": fraction,
        "salt": salt,
        "tolerance": tolerance,          # tracking-within-tol threshold (fixed)
        "heldout_keys": split["heldout"],
        "n_curriculum": len(split["curriculum"]),
        "integrity": _digest_of(split["heldout"], salt),
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        manifest = json.load(f)
    expect = _digest_of(manifest["heldout_keys"], manifest["salt"])
    if manifest.get("integrity") != expect:
        raise ValueError(f"manifest integrity check failed: {path}")
    return manifest


def curriculum_keys(manifest: Dict[str, Any], all_keys: Iterable[str]) -> List[str]:
    """Training-side allowlist: everything not held out (recomputed from the
    hash, so a key added later still lands on its stable side)."""
    held = set(manifest["heldout_keys"])
    salt, fraction = manifest["salt"], manifest["fraction"]
    out = []
    for k in all_keys:
        if k in held:
            continue
        if _key_bucket(k, salt) < fraction:
            continue  # new key that hashes into the held-out side
        out.append(k)
    return sorted(out)


def heldout_commands(manifest: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """The held-out command tuples the eval must FORCE (from the manifest)."""
    return [parse_command_key(k) for k in manifest["heldout_keys"]]


def heldout_record_from_metrics_eval(
    metrics_eval: Dict[str, Any],
    manifest: Dict[str, Any],
    it: int,
    strict: bool = True,
) -> Dict[str, Any]:
    """One held-out eval pass's metrics_eval.json → one protected record.

    Expected keys (produced by eval_rollout.py --heldout):
      - 'heldout_success_rate' (fraction of forced held-out commands tracked
        within tolerance), OR 'success_rate' as a fallback name;
      - 'eval_keys': the command keys the eval actually forced — integrity
        guarded: must be a SUBSET of the manifest's held-out keys, else refuse
        (a mis-wired eval can't silently feed a wrong protected metric)."""
    success = metrics_eval.get("heldout_success_rate")
    if success is None:
        success = metrics_eval.get("success_rate")
    if success is None:
        raise ValueError(
            "metrics_eval has neither 'heldout_success_rate' nor 'success_rate'")
    if strict and isinstance(metrics_eval.get("eval_keys"), list):
        foreign = sorted(set(metrics_eval["eval_keys"]) - set(manifest["heldout_keys"]))
        if foreign:
            raise ValueError(
                f"eval ran outside the held-out grid ({len(foreign)} foreign "
                f"keys, e.g. {foreign[:3]}); refusing to emit the protected metric")
    if not (0.0 <= float(success) <= 1.0):
        raise ValueError(f"heldout_success_rate out of range: {success}")
    record: Dict[str, Any] = {
        "it": it,
        "heldout_success_rate": round(float(success), 6),
        "heldout_n_commands": len(manifest["heldout_keys"]),
        "heldout_manifest_integrity": manifest["integrity"],
    }
    err = metrics_eval.get("mean_tracking_error")
    if isinstance(err, (int, float)):
        record["heldout_mean_tracking_error"] = float(err)
    # forward the per-condition breakdown (the anti-SONIC dominating-condition
    # diagnostic) — it is produced ONLY on the held-out path, so it must ride
    # on the held-out record, not be dropped here (fix-verify medium).
    if isinstance(metrics_eval.get("per_condition"), dict):
        record["heldout_per_condition"] = metrics_eval["per_condition"]
    return record


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Isaac Lab held-out watcher (CPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make-manifest", help="build a command grid + split")
    # in-envelope defaults (achievable within tol) so the metric discriminates
    # policy quality; see command_grid docstring (verified in-container).
    mk.add_argument("--lin-vel-x", default="-1,-0.5,0.5,1",
                    help="comma list of lin_vel_x samples (keep within the "
                         "achievable tracking envelope, e.g. |vx|<=1)")
    mk.add_argument("--ang-vel-z", default="-1,-0.5,0.5,1",
                    help="comma list of ang_vel_z samples (|wz|<=1)")
    mk.add_argument("--lin-vel-y", default="0")
    mk.add_argument("--fraction", type=float, default=0.34)
    mk.add_argument("--salt", required=True)
    mk.add_argument("--tolerance", type=float, default=0.25)
    mk.add_argument("--out", required=True)

    rec = sub.add_parser("record", help="metrics_eval.json -> protected record")
    rec.add_argument("--metrics-eval", required=True)
    rec.add_argument("--manifest", required=True)
    rec.add_argument("--it", type=int, required=True)
    rec.add_argument("--append-to")

    args = p.parse_args(argv)
    if args.cmd == "make-manifest":
        fl = lambda s: [float(x) for x in s.split(",") if x != ""]  # noqa: E731
        keys = command_grid(fl(args.lin_vel_x), fl(args.ang_vel_z), fl(args.lin_vel_y))
        m = write_manifest(args.out, keys, args.fraction, args.salt,
                           tolerance=args.tolerance)
        print(f"manifest -> {args.out} ({len(m['heldout_keys'])} held-out / "
              f"{m['n_curriculum']} curriculum commands)")
    else:
        with open(args.metrics_eval) as f:
            metrics_eval = json.load(f)
        record = heldout_record_from_metrics_eval(
            metrics_eval, load_manifest(args.manifest), args.it)
        line = json.dumps(record)
        if args.append_to:
            with open(args.append_to, "a") as f:
                f.write(line + "\n")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

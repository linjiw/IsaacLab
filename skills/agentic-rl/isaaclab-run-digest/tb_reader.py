# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""TensorBoard scalar → digest-record normalization for Isaac Lab runs.

This is the PURE, CPU-testable half of the run-digest skill: it maps Isaac Lab
TB scalar tags to the digest's train-stream keys and groups them into
per-iteration records. The ACTUAL reading of TB event files happens
in-container (Isaac Lab's kit python has tensorboard; the host may not), driven
by isaaclab-job-adapter._read_train_records, which calls records_from_scalars
here on the {tag: [(step, value), ...]} dump it extracts.

The digest is then assembled by the vendored runmanager/core/digest.py
build_digest() over these records. Nothing here imports an engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Isaac Lab TensorBoard tag → digest train-stream key. The two engine-neutral
# aggregates (Episode/rew_mean, Episode/len_mean) are what the run-manager loop
# itself reads; the per-term prefixes let the digest build its termination /
# episode / curriculum sections. Verified against a real skrl cartpole +
# Anymal-C run's TB tags (2026-07-08/09).
_TAG_ALIASES = {
    # rl_games
    "rewards/step": "Episode/rew_mean",
    "rewards/iter": "Episode/rew_mean",
    "rewards/time": "Episode/rew_mean",
    "episode_lengths/step": "Episode/len_mean",
    "episode_lengths/iter": "Episode/len_mean",
    # skrl (VERIFIED)
    "Reward / Total reward (mean)": "Episode/rew_mean",
    "Reward / Return (mean)": "Episode/rew_mean",
    "Episode / Total timesteps (mean)": "Episode/len_mean",
    # common
    "Train/mean_reward": "Episode/rew_mean",
    "Train/mean_episode_length": "Episode/len_mean",
}
# prefixes carried through verbatim (digest section builders read these)
_PASSTHROUGH_PREFIXES = ("Episode_Reward/", "Episode_Termination/",
                         "Curriculum/", "Metrics/")


def normalize_tag(tag: str) -> Optional[str]:
    """Isaac Lab TB tag → digest key, or None to drop the tag."""
    if tag in _TAG_ALIASES:
        return _TAG_ALIASES[tag]
    for pref in _PASSTHROUGH_PREFIXES:
        if tag.startswith(pref):
            return tag
    return None


def records_from_scalars(scalars: Dict[str, List[tuple]]) -> List[Dict[str, Any]]:
    """Turn a {tag: [(step, value), ...]} scalar dump into per-iteration
    digest train records (one dict per step, keyed by normalized digest key).

    Steps present in any tracked tag become record boundaries; a record
    carries every normalized tag that has a value at that step."""
    by_step: Dict[int, Dict[str, Any]] = {}
    for tag, series in scalars.items():
        key = normalize_tag(tag)
        if key is None:
            continue
        for step, value in series:
            if not isinstance(value, (int, float)):
                continue
            rec = by_step.setdefault(int(step), {})
            rec.setdefault(key, float(value))
    return [{"it": step, **by_step[step]} for step in sorted(by_step)]

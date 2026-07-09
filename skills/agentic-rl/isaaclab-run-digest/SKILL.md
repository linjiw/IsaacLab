---
name: isaaclab-run-digest
description: >-
  Turn raw Isaac Lab training logs into the compact, trend-annotated
  digest.json that is the ONLY thing the curriculum-manager LLM reads — no
  per-step data ever reaches the LLM. Reads TensorBoard scalars (Episode
  rewards, curriculum term states, command-tracking metrics, termination
  fractions) plus eval records, and summarizes each series as
  {last, mean_recent, slope_recent, trend, n_points}. Use when building the
  per-tick observation for the manager, or when inspecting what the manager
  saw at a given tick.
license: Apache-2.0
compatibility: >-
  Pure Python 3.9+ stdlib (no numpy). Reuses the vendored
  runmanager/core/digest.py builder; this skill supplies the Isaac Lab
  TensorBoard reader and the per-framework train_scalar_keys. Reading TB
  event files requires tensorboard/tbparse or the protobuf reader.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- observability
---

# Isaac Lab Run Digest (the manager's eyes)

Produces `digest.json` (schema 0.1.0, from `runmanager/core/digest.py`) — the compact
observation the LLM manager reads each tick. **Nothing per-step reaches the LLM.**

## What the digest contains
- **`eval`** — `success_rate` / `heldout_success_rate` / gate metric, each as
  `{last, mean_recent, slope_recent, trend, n_points}`; plus `failed_keys` diff (newly
  failing / recovered / persistent).
- **`train`** — injected `train_scalar_keys` summarized; per-term `episode_terms_last`,
  `termination_terms_mean_recent` (which axis is binding?), `scheduled_params_last`.
- **`sampler`** — entropy / cap saturation / top-k share (when a sampler stream exists).
- **`knobs`** — current value + `ticks_since_change` per knob.
- **`decision_history`** — recent journal entries with scored outcomes, so the LLM sees what
  its last interventions did.

## Isaac Lab metric mapping (TensorBoard → digest keys)
Isaac Lab merges every manager's log dict into `env.extras["log"]`, written to TensorBoard:

| Isaac Lab scalar | Digest role |
|---|---|
| `Episode_Reward/<term>` | per-term episode reward (`episode_terms_last`) |
| `Episode_Termination/<term>` | termination fraction (`termination_terms_*` — binding axis) |
| `Curriculum/<term>` | curriculum term state (e.g. mean terrain level) |
| `Metrics/<command>/<metric>` | command-tracking error (locomotion gate-metric input) |
| framework reward/length means | `Episode/rew_mean`, `Episode/len_mean` (loop-consumed defaults) |

`train_scalar_keys` is injected per framework — the two engine-neutral defaults
(`Episode/rew_mean`, `Episode/len_mean`) are what the loop itself reads; everything else the
adapter supplies.

## Files
- `tb_reader.py` — the PURE, CPU-testable normalization: `normalize_tag` (Isaac Lab TB tag →
  digest key, verified against real skrl runs) + `records_from_scalars` ({tag: [(step, value)]}
  → per-iteration digest records). This is the single source of truth for the tag mapping;
  `isaaclab-job-adapter` imports it.
- `test_tb_reader.py` — CPU tests for the tag mapping + record grouping.

Note the split: **reading** TB event files happens IN-CONTAINER (Isaac Lab's kit python has
tensorboard; the host may not), driven by `isaaclab-job-adapter._read_train_records`, which
docker-execs a tiny in-container snippet to extract the `{tag: [(step, value)]}` dump and then
calls `records_from_scalars` here. The digest is **assembled** by the vendored
`runmanager/core/digest.py::build_digest` over those records.

## Quick start
```bash
cd skills/agentic-rl/isaaclab-run-digest
python3 -m pytest test_tb_reader.py -q
python3 -c "from tb_reader import normalize_tag; print(normalize_tag('Reward / Total reward (mean)'))"
# -> Episode/rew_mean
```

## Related
- `tb_reader.py` — the tag normalization (this skill).
- `runmanager/core/digest.py` — the builder these records feed (`build_digest`, `summarize_series`).
- `isaaclab-job-adapter` — its `parse_segment` reads TB in-container and calls `records_from_scalars`.
- `isaaclab-curriculum-manager` — the sole consumer of the assembled digest.

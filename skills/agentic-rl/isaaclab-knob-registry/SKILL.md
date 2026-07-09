---
name: isaaclab-knob-registry
description: >-
  The Isaac Lab curriculum-RL action space: a typed whitelist (registry.yaml)
  of the knobs an LLM manager may change — reward weights, terrain difficulty,
  command ranges, domain-randomization ranges, and optimizer meta-parameters —
  each with a hard range, max step, cooldown, and Hydra config path. Anything
  not in the registry is outside the manager's action space by construction.
  Use when defining or auditing which knobs the manager can touch, or when
  validating a proposed knob decision before it is applied.
license: Apache-2.0
compatibility: >-
  Pure Python 3.9+ stdlib + PyYAML (CPU-testable). Reuses the vendored
  runmanager/core/registry.py validator; this skill supplies the Isaac Lab
  registry.yaml content and a thin CLI. Applying knobs live requires the
  isaaclab-job-adapter and a GPU.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- registry
---

# Isaac Lab Knob Registry (the action space)

The **whitelist**. A decision naming a knob not in `registry.yaml`, or exceeding its
`hard_range` / `max_step` / `cooldown_ticks`, is statically rejected **before** any GPU spend
by `runmanager/core/registry.py::KnobRegistry.validate_decision`.

## Knob spec shape
```yaml
meta:
  global_rules: {max_changes_per_tick: 1, default_action: none}
knobs:
  learning_rate:
    family: optimizer            # data_curriculum | schedule | optimizer
    type: float                  # float | choice
    config_path: agent.params.config.learning_rate   # framework-specific Hydra path
    default: 1.0e-3
    hard_range: [1.0e-4, 3.0e-3]
    max_step: {kind: multiplicative, factor: 1.5}     # multiplicative | additive | notch
    cooldown_ticks: 4
    status: available            # available | patch | design
    verified_source: "source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
```

## Families and how they map to Isaac Lab
- **`data_curriculum`** — command/pose ranges, terrain difficulty. `env.commands.*.ranges.*`.
- **`schedule`** — native `CurriculumTermCfg` terms (`modify_reward_weight`, `modify_term_cfg`).
  **Held-out gated:** a `schedule` knob may only fire when a non-null held-out metric exists
  (machine-enforced by the validator's `heldout_gated_families`).
- **`optimizer`** — agent-cfg meta-params. Prefix differs by framework
  (`agent.params.config.*` for rl_games, `agent.*` for skrl).

## `status` semantics
- `available` — mechanism works via a plain Hydra override; safe to apply live.
- `patch` — requires a native `CurriculumTermCfg` to be present in the env cfg (the term is
  the mechanism; the knob tunes its `values`/`num_steps`/`weight`).
- `design` — mechanism not built; replay/simulation only (`allow_design=True`).

## Config-drift verification
`KnobRegistry.verify_against_config(state, resolved_cfg, knob_paths)` reconciles believed knob
values against the run's saved `params/{env,agent}.yaml`, raising `ConfigDriftError` on any
divergence — so a wrong belief can never turn a "one-notch" change into a 2× jump.

## Files (bundled)
- `registry.yaml` — the Isaac Lab action space (authored per task family / framework).
- `validate.py` — thin CLI over the vendored validator.

## Quick start
```bash
cd skills/agentic-rl/isaaclab-knob-registry
python3 validate.py --registry registry.yaml --decision '{"action":"set","knob":"learning_rate","value":0.0008,"rationale":"...","expected_effect":"...","tripwire":{"metric":"eval/gate","drop_pct":10,"evals":2}}'
```

## Related
- `runmanager/core/registry.py` — the validator this configures.
- `isaaclab-job-adapter` — `knob_to_config_path` must cover every `available`/`patch` knob here.
- `isaaclab-curriculum-manager` — emits the decisions this validates.

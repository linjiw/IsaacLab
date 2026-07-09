# runmanager — engine-agnostic LLM-guided curriculum-RL core

This package is the **backend-agnostic control core** for the `agentic-rl` skills: the
outer-loop state machine that supervises a live RL run at checkpoint cadence
(digest → decide → validate → apply → watch), plus its guardrails.

**It contains no IsaacLab (or any engine) imports.** Engine specifics enter through a
6-method `EngineAdapter` and injected `LoopConfig`. See `../DESIGN.md` for the full plan.

## Provenance
Vendored **verbatim** from the `groot-tao-agentic-rl-curriculum` project's
`experiments/run-manager-core/` — a SONIC-decoupled refactor of the Phase-2 SmokeDriver.
Kept byte-identical on purpose (the docstrings cite exact `smoke_driver.py` line numbers as
provenance and the journal format is a serialization contract). Do **not** rewrite these
files; extend via new adapters/policies alongside them.

The one prior-art fact that shaped this whole effort: that project reached a documented
**null result** — an open-loop scripted replay of the manager's knob ladder matched-or-beat
the closed-loop LLM manager. The durable asset is this scaffold, not the adaptivity claim.
Always run a `scripted` arm as the bar the `manager` arm must clear.

## Layout
```
core/
  protocols.py     EngineAdapter (6 methods) + Policy/ObservingPolicy; Segment/ParsedSegment/Decision/Tripwire
  loop.py          RunManager.run() — the 20-step per-segment state machine; LoopConfig; control/scripted factories
  registry.py      KnobRegistry — whitelist + static validator (hard_range/max_step/cooldown/pending-gate/config-drift)
  digest.py        build_digest() — logs → trend-annotated observation (injectable train_scalar_keys)
  tripwire.py      TripwireWatch — rollback when guard metric drops >X% for N evals (abs-floor + stale-eval refusal)
  equivalence.py   EquivalenceGate — "same run?" verdict (window-mean τ; pointwise gating is dead)
  journal.py       byte-compatible decision-journal construction / save / load
adapters/
  mock.py          reference EngineAdapter impls + reference policies (TrainSideBandPolicy=manager, ScriptedPolicy=ablation) + MOCK_REGISTRY_SPEC
tests/             138 CPU tests (135 pass + 3 skip here; the 3 skips are SONIC-journal byte-compat oracles absent from IsaacLab)
```

## Running the tests
Pure-Python (stdlib + PyYAML), Python 3.9+, CPU-only — no GPU, no IsaacLab, no Isaac Sim.

```bash
python3 -m venv /tmp/rmc-venv && /tmp/rmc-venv/bin/python -m pip install -q pytest pyyaml
cd skills/agentic-rl/runmanager && /tmp/rmc-venv/bin/python -m pytest -q
# expected: 135 passed, 3 skipped
```

The 3 skips (`test_journal.py`) are byte-compat regression tests against a real SONIC Phase-2
journal file that is intentionally not vendored into IsaacLab; they skip cleanly when absent.

## What IsaacLab must supply (implemented in the `isaaclab-job-adapter` skill)
1. One `EngineAdapter` for `rl_games`/`skrl` (`launch_segment`, `wait`, `parse_segment`,
   `eval_segment`, `resolved_config_text`, `knob_to_config_path`).
2. A `registry.yaml` action space (the `isaaclab-knob-registry` skill).
3. `train_scalar_keys` + an eval gate metric per task family (the `isaaclab-task-catalog` skill).

The loop, registry validator, tripwire, digest, and equivalence gate carry over unchanged.

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Cross-consistency tests: the task catalog, the knob registry, and the
adapter's knob→Hydra tables must agree. These catch wiring drift (a task
declaring a knob the registry doesn't have, or the registry naming a knob the
adapter can't map to a Hydra path) without any GPU."""

import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTIC = os.path.normpath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_AGENTIC, "runmanager"))
sys.path.insert(0, os.path.join(_AGENTIC, "isaaclab-job-adapter"))

import catalog  # noqa: E402
from core.registry import KnobRegistry  # noqa: E402
from isaaclab_adapter import knob_to_hydra  # noqa: E402

REGISTRY = os.path.join(_AGENTIC, "isaaclab-knob-registry", "registry.yaml")
TASKS = os.path.join(_HERE, "tasks.yaml")


def _registry():
    with open(REGISTRY) as f:
        spec = yaml.safe_load(f)
    fams = (spec.get("meta") or {}).get("heldout_gated_families", ("schedule",))
    return KnobRegistry(spec, heldout_gated_families=tuple(fams))


def test_registry_yaml_is_valid():
    reg = _registry()   # KnobRegistry.__init__ validates every knob spec
    assert reg.knobs, "registry has no knobs"


def test_every_registry_knob_maps_to_a_hydra_path_both_frameworks():
    reg = _registry()
    for fw in ("skrl", "rl_games"):
        table = knob_to_hydra(fw)
        missing = [k for k in reg.knobs if k not in table]
        assert not missing, f"{fw}: registry knobs with no Hydra mapping: {missing}"


def test_every_task_knob_is_in_the_registry():
    reg = _registry()
    tasks = catalog.load_tasks(TASKS)
    for task, spec in tasks.items():
        for knob in spec.get("knob_subset", []):
            assert knob in reg.knobs, f"{task}: knob {knob!r} not in registry"


def test_phase1_locomotion_task_is_ready():
    tasks = catalog.load_tasks(TASKS)
    knobs = catalog.load_registry_knob_names(REGISTRY)
    rep = catalog.readiness(tasks, knobs)
    r = rep["Isaac-Velocity-Rough-Anymal-C-v0"]
    assert r["ready"] is True
    assert not r["unknown_knobs"]
    assert r["gate_metric"] == "tracking_within_tol"


def test_readiness_flags_unknown_knob():
    rep = catalog.readiness(
        {"T": {"gate_metric": {"kind": "success_rate"},
               "knob_subset": ["learning_rate", "bogus_knob"]}},
        ["learning_rate"])
    assert rep["T"]["ready"] is False
    assert rep["T"]["unknown_knobs"] == ["bogus_knob"]


def test_status_not_ready_overrides_structural_check():
    # a well-formed task with all-known knobs but status:not_ready is NOT ready
    rep = catalog.readiness(
        {"T": {"gate_metric": {"kind": "success_rate"},
               "knob_subset": ["learning_rate"], "status": "not_ready"}},
        ["learning_rate"])
    assert rep["T"]["ready"] is False
    assert rep["T"]["status"] == "not_ready"

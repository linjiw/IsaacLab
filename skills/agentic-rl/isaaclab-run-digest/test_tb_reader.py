# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""CPU tests for the run-digest TB-scalar normalization (single source of
truth for Isaac Lab's TB tag → digest-key mapping)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tb_reader import normalize_tag, records_from_scalars  # noqa: E402


def test_real_skrl_tags_map():
    # VERIFIED against real Isaac-Cartpole-v0 / Anymal-C skrl runs
    assert normalize_tag("Reward / Total reward (mean)") == "Episode/rew_mean"
    assert normalize_tag("Episode / Total timesteps (mean)") == "Episode/len_mean"
    assert normalize_tag("Episode_Reward/alive") == "Episode_Reward/alive"
    assert normalize_tag("Curriculum/terrain_levels") == "Curriculum/terrain_levels"


def test_dropped_tags():
    assert normalize_tag("Loss / Policy loss") is None
    assert normalize_tag("Stats / Inference time (ms)") is None
    assert normalize_tag("unmapped/tag") is None


def test_records_grouped_by_step():
    scalars = {
        "Reward / Total reward (mean)": [(0, 1.0), (1, 2.0)],
        "Episode_Termination/base_contact": [(0, 0.3), (1, 0.2)],
        "Loss / Policy loss": [(0, 9.9)],  # dropped
    }
    recs = records_from_scalars(scalars)
    assert [r["it"] for r in recs] == [0, 1]
    assert recs[0]["Episode/rew_mean"] == 1.0
    assert recs[0]["Episode_Termination/base_contact"] == 0.3
    assert "Loss / Policy loss" not in recs[0]


def test_empty():
    assert records_from_scalars({}) == []

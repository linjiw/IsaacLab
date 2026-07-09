# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, The Isaac Lab Project Developers.
"""Isaac Lab curriculum-manager policies (the manager + scripted arms).

The vendored runmanager/adapters/mock.py policies are byte-identical SONIC
references whose knob names don't exist in the Isaac Lab registry. These are
the Isaac Lab analogs.

Design faithfulness (SONIC methodology + review B3/B4/B5):
  - The manager gates on the PROTECTED held-out metric (heldout_success_rate),
    NOT on the eval signal it could otherwise steer per-item. It cannot see
    which held-out commands, reweight them, change their tolerance, or add them
    to training — protection is structural (the frozen grid + fixed tolerance
    live in the watcher's manifest). Hard rule: no held-out metric -> no action.
  - MULTI-KNOB action space (review B5): one change per tick (a hard rule), but
    chosen from more than one knob — command range (primary difficulty lever)
    and learning rate (stability lever on plateau) — so a manager-vs-scripted
    verdict is not a 1-D artifact.
  - The SCRIPTED arm replays the manager's REALIZED decision journal (review
    B4), generically — whatever mix of knobs the manager actually moved — so
    the ON-vs-OFF comparison isolates "does reading the digest help" from
    trajectory differences. It is DERIVED, never hand-authored.

Honest framing (the SONIC null result): the manager must BEAT the open-loop
scripted replay of its own ladder to justify itself.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

HELDOUT_KEY = "heldout_success_rate"


# ── journal -> scripted ladder (review B4) ───────────────────────────
def ladder_from_journal(journal: List[Dict[str, Any]]) -> Dict[int, Tuple[str, Any]]:
    """Extract tick -> (knob, value) from a completed manager run's APPLIED
    decisions. This is the ONLY sanctioned way to build the scripted arm's
    ladder: it walks the manager's exact realized decision stream (knob name
    AND value AND tick), so the scripted arm reproduces the manager's
    trajectory without reading the digest."""
    ladder: Dict[int, Tuple[str, Any]] = {}
    for e in journal:
        if not e.get("applied"):
            continue
        dec = e.get("decision") or {}
        if dec.get("action") == "set" and dec.get("knob") is not None:
            ladder[int(e["tick"])] = (dec["knob"], dec["value"])
    return ladder


# ── manager arm: multi-knob, held-out-gated ──────────────────────────
@dataclasses.dataclass
class LocomotionManagerPolicy:
    """Competence-gated, multi-knob curriculum manager for locomotion tasks.

    Decision signal: the PROTECTED heldout_success_rate (digest.eval.<KEY>).
    One change per tick, priority-ordered:

      1. held-out >= t_high for `sustain` evals, range knob off its ceiling
         -> HARDEN: widen command_range one notch (raise task difficulty).
      2. held-out <= t_low  for `sustain` evals, range knob off its floor
         -> EASE:   narrow command_range one notch.
      3. held-out IN-BAND (plateau) for `sustain` evals AND training reward
         trend flat AND lr off its floor
         -> anneal learning_rate one step (stabilize a plateaued run).
      4. else -> none (the default; interventions are rare by design).

    Hard rule 4 (SONIC): if heldout_success_rate is null / trend unknown,
    emit none — the protected metric is the only sanctioned decision input."""

    range_knob: str = "command_range_lin_vel_x"
    lr_knob: str = "learning_rate"
    t_low: float = 0.50
    t_high: float = 0.85
    sustain: int = 3
    notch: float = 0.25
    lr_factor: float = 1.0 / 1.5      # anneal down within the multiplicative step

    def __post_init__(self):
        self._gate_history: List[float] = []

    def observe(self, digest: Dict[str, Any]) -> None:
        ev = (digest.get("eval") or {}).get(HELDOUT_KEY) or {}
        if ev.get("last") is not None:
            self._gate_history.append(ev["last"])

    def _heldout_ok(self, digest: Dict[str, Any]) -> bool:
        """Hard rule 4: a usable protected metric must exist with a known trend."""
        ev = (digest.get("eval") or {}).get(HELDOUT_KEY) or {}
        return ev.get("last") is not None and ev.get("trend") not in (None, "unknown")

    def _reward_flat(self, digest: Dict[str, Any]) -> bool:
        tr = (digest.get("train") or {}).get("Episode/rew_mean") or {}
        return tr.get("trend") == "flat"

    def _tripwire(self) -> Dict[str, Any]:
        # guard the PROTECTED metric; abs-floor lives in the tripwire machinery
        return {"metric": f"eval/{HELDOUT_KEY}", "drop_pct": 15, "evals": 2}

    def propose(self, digest: Dict[str, Any], state, registry) -> Dict[str, Any]:
        if not self._heldout_ok(digest):
            return {"action": "none",
                    "reason": "hard rule 4: no held-out metric (null/unknown trend)"}
        recent = self._gate_history[-self.sustain:]
        if len(recent) < self.sustain:
            return {"action": "none",
                    "reason": f"held-out history {len(recent)}/{self.sustain} — building"}
        tw = self._tripwire()

        # 1/2: command-range difficulty stepping
        if self.range_knob in registry.knobs:
            cur = float(registry.current_of(self.range_knob, state))
            lo, hi = (float(x) for x in registry.knobs[self.range_knob]["hard_range"])
            if all(v >= self.t_high for v in recent) and cur + self.notch <= hi:
                return self._set(self.range_knob, round(cur + self.notch, 4),
                                 f"held-out {[round(v,2) for v in recent]} >= t_high "
                                 f"{self.t_high} x{self.sustain}; widen (harder)",
                                 "difficulty rises; held-out dips then recovers >= t_low",
                                 self.t_low, tw)
            if all(v <= self.t_low for v in recent) and cur - self.notch >= lo:
                return self._set(self.range_knob, round(cur - self.notch, 4),
                                 f"held-out {[round(v,2) for v in recent]} <= t_low "
                                 f"{self.t_low} x{self.sustain}; narrow (easier)",
                                 "difficulty falls; held-out recovers >= t_low",
                                 self.t_low, tw)

        # 3: plateau -> anneal learning rate for stability
        in_band = all(self.t_low < v < self.t_high for v in recent)
        if (in_band and self._reward_flat(digest) and self.lr_knob in registry.knobs):
            cur = float(registry.current_of(self.lr_knob, state))
            lo, hi = (float(x) for x in registry.knobs[self.lr_knob]["hard_range"])
            target = round(cur * self.lr_factor, 8)
            if target >= lo and abs(target - cur) > 0:
                # the claimed effect is "held-out resumes rising", so score it
                # against the held-out metric clearing t_high — NOT a vacuous
                # rew_mean>=0 (auto-satisfied) check (fix-verify medium).
                return self._set(self.lr_knob, target,
                                 f"held-out plateaued in-band {[round(v,2) for v in recent]} "
                                 f"and reward flat; anneal lr {cur}->{target}",
                                 "training stabilizes; held-out resumes rising to >= t_high",
                                 self.t_high, tw)
        return {"action": "none",
                "reason": f"held-out {[round(v,2) for v in recent]} in band, no lever indicated"}

    def _set(self, knob, value, rationale, expected_effect, effect_floor, tw):
        d = {"action": "set", "knob": knob, "value": value,
             "rationale": rationale, "expected_effect": expected_effect,
             "tripwire": tw}
        # score the effect against the held-out metric clearing `effect_floor`
        # (meaningful, falsifiable) — never an auto-satisfied check.
        if effect_floor is not None:
            d["expected_effect_check"] = {"metric": f"eval/{HELDOUT_KEY}",
                                          "op": ">=", "value": effect_floor}
        return d


# ── scripted arm: replay the manager's realized journal (review B4) ───
@dataclasses.dataclass
class IsaacLabScriptedPolicy:
    """Open-loop replay of a ladder DERIVED from a real manager journal
    (ladder_from_journal). Replays each tick's (knob, value) unconditionally;
    reads the digest ONLY to pick the tripwire metric. Deliberately NO
    observe(). Passing a hand-authored ladder is possible but discouraged —
    the whole point (review B4) is that the competitor walks the manager's
    EXACT realized decisions."""

    ladder: Dict[int, Tuple[str, Any]] = dataclasses.field(default_factory=dict)

    def propose(self, digest: Optional[Dict[str, Any]], state, registry) -> Dict[str, Any]:
        rung = self.ladder.get(state.tick)
        if rung is None:
            return {"action": "none",
                    "reason": f"scripted ladder: no rung at tick {state.tick}"}
        knob, value = rung
        ev = (digest or {}).get("eval") or {}
        # NB: build_eval_section always INSERTS heldout_success_rate (as None
        # when absent), so test the VALUE, not key presence — else the
        # eval/success_rate fallback is dead and the tripwire arms on a None
        # baseline, rejecting every rung (fix-verify HIGH regression).
        def _has(k):
            s = ev.get(k)
            return isinstance(s, dict) and s.get("last") is not None
        if _has(HELDOUT_KEY):
            tw = {"metric": f"eval/{HELDOUT_KEY}", "drop_pct": 15, "evals": 2}
        elif _has("success_rate"):
            tw = {"metric": "eval/success_rate", "drop_pct": 15, "evals": 2}
        else:
            tw = {"metric": "Episode/rew_mean", "drop_pct": 20, "evals": 2}
        return {
            "action": "set", "knob": knob, "value": value,
            "rationale": (f"scripted open-loop replay of the manager's realized "
                          f"journal: rung at tick {state.tick} — no digest input"),
            "expected_effect": "reproduces the manager's knob trajectory without closed-loop decisions",
            "tripwire": tw,
        }


# ── LLM arm: the actual "LLM-guided" policy (shells out to claude -p) ─────
PLAYBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "isaaclab-curriculum-manager", "SKILL.md")


@dataclasses.dataclass
class LLMPolicy:
    """The LLM curriculum manager: shells out to `claude -p` with the
    curriculum-manager playbook (SKILL.md) + the tick's digest, and parses one
    schema-shaped decision. This is the "LLM-guided" arm the project goal names;
    LocomotionManagerPolicy is its deterministic rule-based counterpart (the
    validated core of the same playbook), and the two together let us ask
    whether an LLM's judgment beats the fixed rule AND the scripted replay.

    Fail-safe by design (SONIC LLMPolicy): any subprocess/parse failure degrades
    to `action: none` with the reason recorded, so the loop never crashes on a
    flaky/absent CLI and the journal shows exactly what happened.

    Decisions still pass through KnobRegistry.validate_decision in the loop, so
    a hallucinated knob / out-of-range value is rejected there — the LLM cannot
    escape the whitelist or the bounded steps."""

    model: Optional[str] = None
    timeout_s: int = 180
    t_low: float = 0.50            # passed to the prompt so the LLM knows the band
    t_high: float = 0.85

    def __post_init__(self):
        try:
            with open(PLAYBOOK_PATH) as f:
                self._playbook = f.read()
        except OSError:
            self._playbook = "(playbook unavailable)"

    def _prompt(self, digest: Dict[str, Any]) -> str:
        return (
            "You are the Isaac Lab curriculum manager. Follow this playbook "
            "EXACTLY — its hard rules, tick procedure, and decision table. You "
            f"gate on the PROTECTED held-out metric; band t_low={self.t_low}, "
            f"t_high={self.t_high}.\n\n<playbook>\n" + self._playbook +
            "\n</playbook>\n\nCurrent digest (this tick's only observation):\n\n"
            "```json\n" + json.dumps(digest, indent=1, default=str) + "\n```\n\n"
            "Reply with ONLY one fenced ```json block containing the decision "
            'object: {"action":"none","reason":...} OR {"action":"set",'
            '"knob":...,"value":...,"rationale":...,"expected_effect":...,'
            '"tripwire":{"metric":...,"drop_pct":...,"evals":...}}. '
            "The knob MUST be one from the playbook's registry; steps are "
            "bounded by the registry (the harness will reject violations). "
            "No text outside the fenced block."
        )

    def propose(self, digest: Dict[str, Any], state, registry) -> Dict[str, Any]:
        cmd = ["claude", "-p", "--output-format", "text"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = subprocess.run(cmd, input=self._prompt(digest),
                                  capture_output=True, text=True,
                                  timeout=self.timeout_s)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {"action": "none", "reason": f"llm unavailable: {e}"}
        if proc.returncode != 0:
            return {"action": "none",
                    "reason": f"llm error (rc={proc.returncode}): {proc.stderr[:200]}"}
        return self._parse(proc.stdout)

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = m.group(1) if m else text.strip()
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            return {"action": "none",
                    "reason": f"unparseable llm output: {text[:200]!r}"}
        if not isinstance(decision, dict) or "action" not in decision:
            return {"action": "none", "reason": "llm output missing 'action'"}
        return decision

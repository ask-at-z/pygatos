"""Regression tests for correctness bugs found by the provenance-completeness audit.

Each test pins a bug that silently corrupted run metadata or results:
  1. temperature/max_tokens of 0 were coerced to defaults by `or` (temp 0.0 -> 0.7).
  2. No sampling seed ever reached the LLM backend.
  3. Codes whose name already existed were dropped silently (Code.__eq__ is name-only).
  4. Theme.source_cluster was never set, so the inducing partition was unrecoverable.
  5. Application judgment failures returned [] — indistinguishable from "no codes apply".
"""

from pygatos.llm.ollama import OllamaBackend
from pygatos.config import LLMConfig
from pygatos.core.codebook import Code, Theme, Codebook
from pygatos.application.code_applier import PointApplicationResult


class TestSamplingOptions:
    def test_temperature_zero_is_not_coerced_to_default(self):
        b = OllamaBackend(temperature=0.7)
        assert b._options(0.0, None)["temperature"] == 0.0

    def test_max_tokens_zero_is_not_coerced_to_default(self):
        b = OllamaBackend(max_tokens=2048)
        assert b._options(None, 0)["num_predict"] == 0

    def test_none_falls_back_to_defaults(self):
        b = OllamaBackend(temperature=0.7, max_tokens=2048)
        o = b._options(None, None)
        assert o["temperature"] == 0.7 and o["num_predict"] == 2048

    def test_seed_forwarded_when_set_and_absent_when_not(self):
        assert OllamaBackend(seed=42)._options(0.0, None)["seed"] == 42
        assert "seed" not in OllamaBackend()._options(0.0, None)

    def test_seed_propagates_from_config(self):
        cfg = LLMConfig()
        cfg.seed = 7
        assert OllamaBackend.from_config(cfg).seed == 7

    def test_init_side_effects_still_applied(self):
        # guards against the _options refactor orphaning __init__ tail assignments
        b = OllamaBackend()
        assert b.think is False and b.json_max_retries == 3


class TestDroppedCodesAreRecorded:
    def test_duplicate_name_is_recorded_not_silent(self):
        cb = Codebook()
        cb.add_code(Code(name="X", definition="first"), accepted=True)
        cb.add_code(Code(name="X", definition="second, different"), accepted=True)

        assert len(cb.accepted_codes) == 1              # dedup behavior preserved
        assert len(cb.dropped_codes) == 1               # but no longer silent
        d = cb.dropped_codes[0]
        assert d["name"] == "X"
        assert d["definition"] == "second, different"
        assert d["kept_definition"] == "first"
        assert d["reason"] == "duplicate_name"
        assert d["would_be_accepted"] is True

    def test_counts_reconcile(self):
        cb = Codebook()
        evaluated = [Code(name="A", definition="a"), Code(name="B", definition="b"),
                     Code(name="A", definition="dup")]
        for c in evaluated:
            cb.add_code(c, accepted=True)
        assert len(cb.accepted_codes) + len(cb.rejected_codes) + len(cb.dropped_codes) == len(evaluated)

    def test_dropped_codes_survive_serialization(self):
        cb = Codebook()
        cb.add_code(Code(name="X", definition="first"), accepted=True)
        cb.add_code(Code(name="X", definition="second"), accepted=True)
        cb2 = Codebook.from_dict(cb.to_dict())
        assert len(cb2.dropped_codes) == 1
        assert cb2.dropped_codes[0]["definition"] == "second"

    def test_unique_names_are_not_dropped(self):
        cb = Codebook()
        cb.add_code(Code(name="A", definition="a"), accepted=True)
        cb.add_code(Code(name="B", definition="b"), accepted=True)
        assert len(cb.accepted_codes) == 2 and cb.dropped_codes == []


class TestThemeSourceCluster:
    def test_theme_accepts_source_cluster(self):
        t = Theme(name="T", definition="d", codes=[], source_cluster=3)
        assert t.source_cluster == 3
        assert Theme.from_dict(t.to_dict()).source_cluster == 3


class TestApplicationFailureIsDistinguishable:
    def test_failure_flagged(self):
        r = PointApplicationResult(
            information_point="p", source_text="s", applied_codes=[], candidate_codes=[],
            analysis="__JUDGMENT_FAILED__: ValueError: bad json")
        assert r.judgment_failed is True

    def test_genuine_no_codes_not_flagged(self):
        r = PointApplicationResult(
            information_point="p", source_text="s", applied_codes=[], candidate_codes=[],
            analysis="None of the candidate codes clearly apply.")
        assert r.judgment_failed is False

    def test_missing_analysis_not_flagged(self):
        r = PointApplicationResult(information_point="p", source_text="s",
                                   applied_codes=[], candidate_codes=[])
        assert r.judgment_failed is False

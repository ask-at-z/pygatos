"""Tests for the LLM-judged duplicate-rate diagnostic (single-judge D17 rule)."""

import hashlib

import numpy as np
import pytest

from pygatos.diagnostics import compute_duplicate_rate
from pygatos.diagnostics.duplicate_rate import (
    DUPLICATE_JUDGE_SYSTEM, DUPLICATE_JUDGE_PROMPT, TOP_K, MIN_COS, BATCH, top_pairs,
)
from pygatos.llm.base import BaseLLM


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeJudge(BaseLLM):
    """Returns scripted verdict batches; records every prompt it saw."""

    def __init__(self, verdict_batches):
        self._batches = list(verdict_batches)
        self.prompts = []

    def generate(self, prompt, system=None, temperature=None, max_tokens=None):
        raise NotImplementedError

    def generate_json(self, prompt, system=None, temperature=None, max_tokens=None):
        self.prompts.append({"prompt": prompt, "system": system, "temperature": temperature})
        out = self._batches.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    @property
    def model_name(self):
        return "fake-judge"


class FakeEmbedder:
    """Deterministic embeddings: identical texts embed identically."""

    def __init__(self, vectors):
        self._vectors = vectors  # text -> vector

    def embed(self, texts):
        return np.array([self._vectors[t] for t in texts], dtype=float)


def _codes(*names_defs):
    return [{"name": n, "definition": d} for n, d in names_defs]


class TestInstrumentPins:
    """The prompt and constants ARE the metric; validated reference points assume them."""

    def test_judge_system_pinned(self):
        assert _sha(DUPLICATE_JUDGE_SYSTEM) == (
            '6acea36edf8f42c46df19b2af8cf48874c0b998a14589679f4e4f77cd9f9a9c5')

    def test_judge_prompt_pinned(self):
        assert _sha(DUPLICATE_JUDGE_PROMPT) == (
            '94c71f3e3eda13db78e1351e110bf90e323e75b21dd7f9301451fa30b4f8bfdb')

    def test_constants(self):
        assert (TOP_K, MIN_COS, BATCH) == (20, 0.60, 10)


class TestTopPairs:
    def test_floor_excludes_dissimilar_pairs(self):
        codes = _codes(("A", "a"), ("B", "b"), ("C", "c"))
        v = {"A: a": [1.0, 0.0], "B: b": [0.95, 0.312249899919920], "C: c": [0.0, 1.0]}
        # cos(A,B) ~ 0.95; cos(A,C)=0; cos(B,C)~0.31 -- only A-B is above 0.60
        pairs, n_above = top_pairs(codes, FakeEmbedder(v))
        assert len(pairs) == 1 and n_above == 1
        i, j, s = pairs[0]
        assert {codes[i]["name"], codes[j]["name"]} == {"A", "B"}
        assert s == pytest.approx(0.95, abs=1e-6)

    def test_fewer_than_two_codes(self):
        assert top_pairs(_codes(("A", "a")), FakeEmbedder({})) == ([], 0)


class TestComputeDuplicateRate:
    def _setup_three_similar(self):
        codes = _codes(("A", "a"), ("B", "b"), ("C", "c"))
        # all three mutually similar (cos > 0.9): 3 pairs
        v = {"A: a": [1.0, 0.0], "B: b": [0.99, 0.14106735979665883], "C: c": [0.98, 0.19899748742132399]}
        return codes, FakeEmbedder(v)

    def test_rates_and_counts(self):
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([{"verdicts": [
            {"n": 1, "label": "DUPLICATE"}, {"n": 2, "label": "SUBTHEME"}, {"n": 3, "label": "DISTINCT"},
        ]}])
        res = compute_duplicate_rate(codes, judge, emb)
        assert res["n_pairs_judged"] == 3
        assert res["dupe_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert res["subtheme_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert res["distinct_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert res["n_dupe_pairs"] == 1
        assert res["dupe_pairs_per_code"] == pytest.approx(1 / 3, abs=1e-3)
        assert res["rule"] == "single-judge"
        assert res["judge_model"] == "fake-judge"
        # per-pair detail is stored so the rule is a free re-aggregation later
        assert sorted(p["label"] for p in res["pairs"]) == ["DISTINCT", "DUPLICATE", "SUBTHEME"]

    def test_judge_called_at_temperature_zero_with_pinned_prompts(self):
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([{"verdicts": [{"n": n, "label": "DISTINCT"} for n in (1, 2, 3)]}])
        compute_duplicate_rate(codes, judge, emb)
        call = judge.prompts[0]
        assert call["temperature"] == 0.0
        assert call["system"] == DUPLICATE_JUDGE_SYSTEM
        assert "PAIR 1" in call["prompt"] and "PAIR 3" in call["prompt"]

    def test_failed_batch_leaves_pairs_unjudged_not_mislabelled(self):
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([RuntimeError("boom")])
        res = compute_duplicate_rate(codes, judge, emb)
        assert res["n_pairs_judged"] == 0
        assert res["n_pairs_unjudged"] == 3
        assert res["dupe_rate"] == 0.0
        assert all(p["label"] == "UNJUDGED" for p in res["pairs"])

    def test_off_scale_labels_are_unjudged(self):
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([{"verdicts": [
            {"n": 1, "label": "DUPLICATE"}, {"n": 2, "label": "MAYBE"}, {"n": 3, "label": "DISTINCT"},
        ]}])
        res = compute_duplicate_rate(codes, judge, emb)
        assert res["n_pairs_judged"] == 2
        assert res["n_pairs_unjudged"] == 1
        assert res["dupe_rate"] == 0.5

    def test_batching_alignment(self):
        # 3 pairs with batch=2 -> two calls; second batch numbering restarts at 1
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([
            {"verdicts": [{"n": 1, "label": "DUPLICATE"}, {"n": 2, "label": "DISTINCT"}]},
            {"verdicts": [{"n": 1, "label": "SUBTHEME"}]},
        ])
        res = compute_duplicate_rate(codes, judge, emb, batch=2)
        assert len(judge.prompts) == 2
        assert res["n_pairs_judged"] == 3
        assert res["n_dupe_pairs"] == 1

    def test_no_pairs_above_floor(self):
        codes = _codes(("A", "a"), ("C", "c"))
        v = {"A: a": [1.0, 0.0], "C: c": [0.0, 1.0]}
        judge = FakeJudge([])
        res = compute_duplicate_rate(codes, judge, FakeEmbedder(v))
        assert res["n_pairs_judged"] == 0 and res["pairs"] == []
        assert judge.prompts == []  # no call spent

    def test_same_judge_as_generator_warns(self, caplog):
        codes, emb = self._setup_three_similar()
        judge = FakeJudge([{"verdicts": [{"n": n, "label": "DISTINCT"} for n in (1, 2, 3)]}])
        import logging
        with caplog.at_level(logging.WARNING):
            compute_duplicate_rate(codes, judge, emb, generator_model="fake-judge")
        assert any("shares its biases" in r.message for r in caplog.records)

    def test_accepts_code_objects(self):
        from pygatos.core.codebook import Code
        codes = [Code(name="A", definition="a"), Code(name="B", definition="b")]
        v = {"A: a": [1.0, 0.0], "B: b": [0.99, 0.14106735979665883]}
        judge = FakeJudge([{"verdicts": [{"n": 1, "label": "DUPLICATE"}]}])
        res = compute_duplicate_rate(codes, judge, FakeEmbedder(v))
        assert res["dupe_rate"] == 1.0

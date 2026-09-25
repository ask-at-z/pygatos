"""Stage-2 novelty PANEL: majority vote, error handling, and single-judge unchanged.

The novelty gate is the only place in the pipeline whose decisions are irreversible (an accepted
code changes what every later candidate is judged against), and until now one model judged its own
proposals. These tests pin the panel's contract.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from pygatos.core.codebook import Code, Codebook
from pygatos.generation.novelty_evaluator import NoveltyEvaluator


class FakeEmbedder:
    """Deterministic embeddings: identical text -> identical vector, others near-orthogonal."""

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=16)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def judge(verdict, reasoning="r", name="j", raises=False, payload=None):
    m = MagicMock()
    m.model_name = name
    if raises:
        m.generate_json.side_effect = RuntimeError("judge exploded")
    else:
        m.generate_json.return_value = (payload if payload is not None
                                        else {"is_novel": verdict, "reasoning": reasoning,
                                              "similar_to": None})
    return m


def book_with_one():
    cb = Codebook()
    cb.add_code(Code(name="Existing", definition="An existing code."), accepted=True)
    return cb


def evaluate(judges, **kw):
    ev = NoveltyEvaluator(llm=judge(True, name="generator"), embedder=FakeEmbedder(),
                          judges=judges, similarity_threshold=1.01,  # force every verdict to stage 2
                          policy="keep-unless-duplicate", **kw)
    return ev.evaluate(Code(name="Candidate", definition="A candidate code."), book_with_one())


def test_majority_accept():
    res = evaluate([judge(True, name="a"), judge(True, name="b"), judge(False, name="c")])
    assert res.is_novel is True and res.stage == "stage2_accept"
    assert [v["is_novel"] for v in res.panel] == [True, True, False]
    assert {v["model"] for v in res.panel} == {"a", "b", "c"}


def test_majority_reject():
    res = evaluate([judge(False, name="a"), judge(False, name="b"), judge(True, name="c")])
    assert res.is_novel is False and res.stage == "stage2_reject"


def test_failed_judge_is_excluded_not_counted():
    # 1 usable YES + 1 failure -> majority of the usable verdicts = accept
    res = evaluate([judge(True, name="a"), judge(None, name="b", raises=True)])
    assert res.is_novel is True
    assert [v["is_novel"] for v in res.panel] == [True, None]


def test_all_judges_fail_is_an_error_not_an_accept():
    res = evaluate([judge(None, name="a", raises=True), judge(None, name="b", raises=True)])
    assert res.is_novel is False and res.stage == "stage2_error"


def test_tie_rejects_conservatively():
    res = evaluate([judge(True, name="a"), judge(False, name="b")])
    assert res.is_novel is False


def test_unusable_payload_is_not_a_vote():
    res = evaluate([judge(None, name="a", payload={"reasoning": "forgot the verdict"}),
                    judge(True, name="b")])
    assert [v["is_novel"] for v in res.panel] == [None, True]
    assert res.is_novel is True


def test_single_judge_path_unchanged_and_panel_is_none():
    ev = NoveltyEvaluator(llm=judge(True, name="solo"), embedder=FakeEmbedder(),
                          similarity_threshold=1.01, policy="keep-unless-duplicate")
    res = ev.evaluate(Code(name="Candidate", definition="A candidate code."), book_with_one())
    assert res.is_novel is True and res.panel is None


def test_every_judge_sees_the_same_prompt():
    a, b = judge(True, name="a"), judge(True, name="b")
    evaluate([a, b])
    pa = a.generate_json.call_args.kwargs["prompt"]
    pb = b.generate_json.call_args.kwargs["prompt"]
    assert pa == pb and "Candidate" in pa


def test_prepopulated_codebook_without_embeddings_is_visible_to_the_gate():
    """A codebook built from saved codes (no embeddings) must not look empty to the judge."""
    judge_ = judge(False, name="a")
    ev = NoveltyEvaluator(llm=judge_, embedder=FakeEmbedder(), similarity_threshold=1.01,
                          policy="keep-unless-duplicate")
    cb = Codebook()
    cb.add_code(Code(name="Existing", definition="An existing code."), accepted=True)
    assert cb.accepted_codes[0].embedding is None          # the state that used to be skipped
    ev.evaluate(Code(name="Candidate", definition="A candidate code."), cb)
    prompt = judge_.generate_json.call_args.kwargs["prompt"]
    assert "No similar codes in codebook yet" not in prompt
    assert "Existing" in prompt

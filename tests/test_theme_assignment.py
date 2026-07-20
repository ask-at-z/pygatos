"""Tests for per-code theme (re)assignment (ThemeGenerator.assign_codes_to_themes)."""

import numpy as np

from pygatos.core.codebook import Code, Theme, Codebook
from pygatos.generation.theme_generator import ThemeGenerator


class StubLLM:
    """Returns a fixed best_theme and records how many times it was asked."""

    def __init__(self, best_theme):
        self.best_theme = best_theme
        self.calls = 0

    def generate_json(self, prompt, system=None, temperature=None, max_tokens=None):
        self.calls += 1
        return {"best_theme": self.best_theme, "reasoning": "stub"}


class ExplodingEmbedder:
    """Embeddings are all preset in these tests, so embed() must never be called."""

    def embed(self, texts, show_progress=False):  # pragma: no cover
        raise AssertionError("embedder should not be called when embeddings are preset")


def _codebook():
    cb = Codebook()
    themes = [
        Theme(name="Animals", definition="living creatures", embedding=np.array([1.0, 0.0])),
        Theme(name="Vehicles", definition="machines for transport", embedding=np.array([0.0, 1.0])),
    ]
    for t in themes:
        cb.add_theme(t)
    codes = [
        # clearly Animals but mis-filed under Vehicles -> should move (embedding, no LLM)
        Code(name="Dog", definition="a domestic animal", embedding=np.array([1.0, 0.0]), theme="Vehicles"),
        # clearly Animals, already correct -> no move, no LLM
        Code(name="Cat", definition="a feline animal", embedding=np.array([0.95, 0.05]), theme="Animals"),
        # ambiguous (45 deg) -> below threshold -> LLM decides
        Code(name="Amphibious car", definition="drives and swims",
             embedding=np.array([0.7, 0.7]), theme="Animals"),
    ]
    for c in codes:
        cb.add_code(c, accepted=True)
    return cb


def test_low_confidence_routes_to_llm_and_reassigns():
    cb = _codebook()
    llm = StubLLM(best_theme="Vehicles")
    tg = ThemeGenerator(llm=llm, embedder=ExplodingEmbedder())

    report = tg.assign_codes_to_themes(cb, top_k=2, confidence_threshold=0.8, validate=True)

    moves = {m["code"]: m for m in report["moved"]}
    # Dog: high-confidence embedding move, no LLM
    assert moves["Dog"]["to"] == "Animals"
    assert moves["Dog"]["method"] == "embedding"
    # Amphibious car: ambiguous -> LLM picked Vehicles
    assert moves["Amphibious car"]["to"] == "Vehicles"
    assert moves["Amphibious car"]["method"] == "llm"
    # Cat unchanged
    assert "Cat" not in moves
    # exactly one LLM call (only the ambiguous code)
    assert report["n_llm_calls"] == 1
    assert llm.calls == 1
    # membership rebuilt to match code.theme
    sizes = report["theme_sizes_after"]
    assert sizes["Animals"] == 2 and sizes["Vehicles"] == 1
    by_name = {c.name: c.theme for c in cb.accepted_codes}
    assert by_name == {"Dog": "Animals", "Cat": "Animals", "Amphibious car": "Vehicles"}


def test_validate_false_uses_pure_embedding_and_no_llm():
    cb = _codebook()
    llm = StubLLM(best_theme="Vehicles")
    tg = ThemeGenerator(llm=llm, embedder=ExplodingEmbedder())

    report = tg.assign_codes_to_themes(cb, top_k=2, confidence_threshold=0.8, validate=False)

    assert report["n_llm_calls"] == 0 and llm.calls == 0
    # Amphibious car at 45deg ties; argmax picks the first theme (Animals) -> stays, no move
    by_name = {c.name: c.theme for c in cb.accepted_codes}
    assert by_name["Dog"] == "Animals"
    assert by_name["Amphibious car"] == "Animals"


def test_none_answer_keeps_embedding_top_match():
    cb = _codebook()
    llm = StubLLM(best_theme="NONE")
    tg = ThemeGenerator(llm=llm, embedder=ExplodingEmbedder())

    report = tg.assign_codes_to_themes(cb, top_k=2, confidence_threshold=0.8, validate=True)

    # Amphibious car got a NONE -> falls back to embedding top match (Animals), so no move
    moves = {m["code"]: m for m in report["moved"]}
    assert "Amphibious car" not in moves
    assert report["n_llm_calls"] == 1

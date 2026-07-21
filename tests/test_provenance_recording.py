"""Tests for LLM call recording and environment/corpus provenance capture."""

import json

import pandas as pd
import pytest

from pygatos.provenance import (
    LLMCallRecorder, stage, item, current_stage,
    sha256_text, sha256_file, environment_metadata, git_metadata, corpus_metadata,
)
from pygatos.llm.ollama import OllamaBackend


class FakeBackend(OllamaBackend):
    """OllamaBackend with the HTTP layer replaced; keeps the real generate_json retry loop."""

    def __init__(self, responses, **kw):
        super().__init__(**kw)
        self._responses = list(responses)
        self.calls = []

    def generate(self, prompt, system=None, temperature=None, max_tokens=None, **kw):
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        out = self._responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _records(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_records_one_line_per_call(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["hello"])
    with LLMCallRecorder(p) as rec:
        rec.attach(b)
        assert b.generate("prompt-1", system="sys-1", temperature=0.0) == "hello"
    r = _records(p)
    assert len(r) == 1
    assert r[0]["prompt"] == "prompt-1" and r[0]["system_prompt"] == "sys-1"
    assert r[0]["response"] == "hello"
    assert r[0]["requested_temperature"] == 0.0
    # resolved options are what was actually sent
    assert r[0]["options_sent"]["temperature"] == 0.0
    assert r[0]["latency_ms"] >= 0


def test_captures_generate_json_retries_as_separate_records(tmp_path):
    """The retry loop lives inside generate_json; patching the instance's generate must see it."""
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["not json at all", '{"ok": true}'])
    with LLMCallRecorder(p) as rec:
        rec.attach(b)
        out = b.generate_json("give me json", system="be terse")
    assert out == {"ok": True}
    r = _records(p)
    assert len(r) == 2, "retry attempt was not captured"
    # the escalated retry prompt differs from the first, and both are preserved verbatim
    assert r[0]["system_prompt"] != r[1]["system_prompt"]
    assert "CRITICAL" in r[1]["system_prompt"]
    assert r[0]["response"] == "not json at all"


def test_records_errors_and_reraises(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend([RuntimeError("boom")])
    with LLMCallRecorder(p) as rec:
        rec.attach(b)
        with pytest.raises(RuntimeError):
            b.generate("x")
    r = _records(p)
    assert len(r) == 1
    assert "RuntimeError: boom" in r[0]["error"]
    assert r[0]["response"] is None


def test_stage_and_item_tagging(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["a", "b"])
    with LLMCallRecorder(p) as rec:
        rec.attach(b)
        with stage("summarization", item="essay_1.txt"):
            b.generate("p1")
            with item("essay_2.txt"):
                b.generate("p2")
    r = _records(p)
    assert [x["stage"] for x in r] == ["summarization", "summarization"]
    assert [x["item"] for x in r] == ["essay_1.txt", "essay_2.txt"]
    assert current_stage() == (None, None), "context did not reset"


def test_detach_restores_original(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["a", "b"])
    rec = LLMCallRecorder(p)
    rec.attach(b)
    b.generate("logged")
    rec.close()
    b.generate("not logged")
    assert len(_records(p)) == 1


def test_redaction_keeps_hashes_only(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["secret response"])
    with LLMCallRecorder(p, redact_prompts=True) as rec:
        rec.attach(b)
        b.generate("secret prompt", system="secret system")
    r = _records(p)[0]
    assert r["prompt"] is None and r["response"] is None and r["system_prompt"] is None
    assert r["prompt_sha256"] == sha256_text("secret prompt")
    assert r["response_sha256"] == sha256_text("secret response")


def test_recorder_never_breaks_the_run_on_write_failure(tmp_path):
    p = tmp_path / "llm_calls.jsonl"
    b = FakeBackend(["ok"])
    rec = LLMCallRecorder(p)
    rec.attach(b)
    rec._fh.close()  # force write failures
    assert b.generate("still works") == "ok"
    rec.detach()


class TestMetadata:
    def test_environment_metadata(self):
        env = environment_metadata()
        assert env["python"] and env["platform"]
        assert "numpy" in env["packages"]

    def test_git_metadata_reports_dirty_flag(self, tmp_path):
        import subprocess
        from pygatos.provenance import git_metadata as gm
        g = gm("/Users/andrewkatz/Documents/ak_dev/projects/pygatos")
        assert g["sha"] is not None
        assert g["dirty"] in (True, False)

    def test_sha256_file_roundtrip(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello")
        assert sha256_file(f) == sha256_text("hello")
        assert sha256_file(tmp_path / "missing") is None

    def test_corpus_metadata(self, tmp_path):
        src = tmp_path / "corpus.csv"
        df_all = pd.DataFrame({"id": ["a", "b", "c"], "text": ["one two", "three", ""]})
        df_all.to_csv(src, index=False)
        df = df_all[df_all["text"].str.strip() != ""].reset_index(drop=True)
        m = corpus_metadata(df, "text", "id", src, n_raw=len(df_all),
                            filter_description="drop empty text")
        assert m["n_raw_rows"] == 3 and m["n_analyzed"] == 2 and m["n_excluded"] == 1
        assert m["analyzed_ids"] == ["a", "b"] and m["ids_unique"] is True
        assert m["source_sha256"] and m["content_sha256_order_independent"]
        assert m["word_count"]["total"] == 3

    def test_corpus_hash_is_order_independent_but_content_sensitive(self, tmp_path):
        src = tmp_path / "c.csv"
        src.write_text("x")
        d1 = pd.DataFrame({"id": ["a", "b"], "text": ["one", "two"]})
        d2 = pd.DataFrame({"id": ["b", "a"], "text": ["two", "one"]})
        d3 = pd.DataFrame({"id": ["a", "b"], "text": ["one", "CHANGED"]})
        h = lambda d: corpus_metadata(d, "text", "id", src, len(d))["content_sha256_order_independent"]
        assert h(d1) == h(d2), "row order should not change the content hash"
        assert h(d1) != h(d3), "content change must change the hash"

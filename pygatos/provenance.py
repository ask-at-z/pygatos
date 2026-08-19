"""Run provenance: raw LLM call recording and environment capture.

Everything here exists because a GATOS run is **not reproducible by re-running it**. LLM sampling
is unseeded by default and is not byte-deterministic even at temperature 0 on a local Ollama
stack, and re-embedding is not bit-identical across processes/devices. Any intermediate state
that is not written at run time is therefore gone permanently. These helpers capture the parts a
reviewer needs to audit a run they cannot reproduce.

Two pieces:

- :class:`LLMCallRecorder` — appends one JSONL record per **HTTP attempt** to the model. It patches
  the backend *instance's* ``generate`` method, so the retry loop inside ``generate_json`` (which
  calls ``self.generate`` with an escalated system prompt) is captured as separate records rather
  than hidden behind a single logical call.
- environment/model/git/corpus metadata functions, for the run manifest.

Recorded prompts and responses contain the source text that was sent to the model, so the call log
inherits the confidentiality of the corpus.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Stage/item context so each recorded call can be attributed to a pipeline phase and the item
# (document id, cluster id, code name, information-point index) being processed.
_stage_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("pygatos_stage", default=None)
_item_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("pygatos_item", default=None)


@contextlib.contextmanager
def stage(name: str, item: Any = None):
    """Tag every LLM call made inside this block with a pipeline stage (and optional item)."""
    t_stage = _stage_var.set(name)
    t_item = _item_var.set(item)
    try:
        yield
    finally:
        _stage_var.reset(t_stage)
        _item_var.reset(t_item)


@contextlib.contextmanager
def item(value: Any):
    """Tag calls with the specific item being processed, keeping the enclosing stage."""
    tok = _item_var.set(value)
    try:
        yield
    finally:
        _item_var.reset(tok)


def current_stage() -> tuple[Optional[str], Optional[Any]]:
    return _stage_var.get(), _item_var.get()


class LLMCallRecorder:
    """Append one JSONL record per LLM HTTP attempt.

    Usage::

        rec = LLMCallRecorder(path)
        rec.attach(pipeline.llm)
        ...
        rec.close()
    """

    def __init__(self, path: str | Path, redact_prompts: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self.redact_prompts = redact_prompts
        self.n_calls = 0
        self.n_errors = 0
        self._detached: list = []

    def attach(self, backend) -> "LLMCallRecorder":
        """Patch ``backend.generate`` on the instance so retries inside generate_json are seen."""
        original = backend.generate
        recorder = self

        def recording_generate(prompt, system=None, temperature=None, max_tokens=None, **kwargs):
            stage_name, item_val = current_stage()
            # Resolve the options actually sent, when the backend exposes them.
            try:
                opts = backend._options(temperature, max_tokens)
            except Exception:
                opts = {"temperature": temperature, "num_predict": max_tokens}
            t0 = time.perf_counter()
            err = None
            response = None
            try:
                response = original(prompt, system=system, temperature=temperature,
                                    max_tokens=max_tokens, **kwargs)
                return response
            except Exception as e:  # record the failure, then let it propagate unchanged
                err = f"{type(e).__name__}: {e}"
                recorder.n_errors += 1
                raise
            finally:
                recorder.n_calls += 1
                rec = {
                    "call_id": recorder.n_calls,
                    "stage": stage_name,
                    "item": item_val,
                    "model": getattr(backend, "model_name", None),
                    "options_sent": opts,
                    "requested_temperature": temperature,
                    "requested_max_tokens": max_tokens,
                    "system_prompt": None if recorder.redact_prompts else system,
                    "prompt": None if recorder.redact_prompts else prompt,
                    "response": None if recorder.redact_prompts else response,
                    "system_prompt_sha256": sha256_text(system) if system else None,
                    "prompt_sha256": sha256_text(prompt),
                    "response_sha256": sha256_text(response) if response is not None else None,
                    "response_chars": len(response) if isinstance(response, str) else None,
                    "error": err,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
                recorder._write(rec)

        backend.generate = recording_generate  # instance attribute shadows the class method
        self._detached.append((backend, original))
        return self

    def detach(self) -> None:
        """Restore original generate methods."""
        for backend, original in self._detached:
            try:
                del backend.generate
            except AttributeError:
                backend.generate = original
        self._detached.clear()

    def _write(self, rec: dict) -> None:
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception as e:  # never let provenance logging break a run
            logger.warning(f"LLMCallRecorder failed to write record: {e}")

    def stats(self) -> dict:
        return {"n_calls": self.n_calls, "n_errors": self.n_errors, "path": str(self.path)}

    def close(self) -> None:
        self.detach()
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------------------
# Metadata capture
# --------------------------------------------------------------------------------------

def sha256_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_VERSION_PACKAGES = [
    "numpy", "scipy", "scikit-learn", "pandas", "umap-learn", "numba", "llvmlite",
    "pynndescent", "hdbscan", "sentence-transformers", "transformers", "torch", "requests",
]


def environment_metadata() -> dict:
    """Python/platform/library versions needed to reconstruct the numerical stack."""
    from importlib.metadata import version, PackageNotFoundError

    versions = {}
    for pkg in _VERSION_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
        except Exception as e:
            versions[pkg] = f"<error: {e}>"
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }


def git_metadata(repo_path: str | Path) -> dict:
    """Commit SHA plus a dirty flag — a SHA alone is misleading when the tree has edits."""
    repo_path = str(repo_path)

    def _run(args):
        try:
            return subprocess.check_output(["git", "-C", repo_path] + args, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    sha = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return {
        "sha": sha,
        "dirty": dirty,
        "dirty_files": [l[3:] for l in status.splitlines()] if status else [],
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["describe", "--always", "--dirty"]),
    }


def ollama_metadata(base_url: str, model: str, timeout: int = 10) -> dict:
    """Ollama server version + the model's blob digest and quantization.

    A model *tag* is mutable — it can be re-pulled and point at different weights. The digest is
    what actually identifies the weights used.
    """
    import requests

    out: dict = {"base_url": base_url, "model_tag": model, "server_version": None,
                 "digest": None, "quantization_level": None, "parameter_size": None,
                 "family": None, "error": None}
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout)
        if r.ok:
            out["server_version"] = r.json().get("version")
    except Exception as e:
        out["error"] = f"version: {type(e).__name__}: {e}"
    # digest lives in /api/tags; details live in /api/show
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if r.ok:
            for m in r.json().get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    out["digest"] = m.get("digest")
                    det = m.get("details") or {}
                    out["quantization_level"] = det.get("quantization_level")
                    out["parameter_size"] = det.get("parameter_size")
                    out["family"] = det.get("family")
                    break
    except Exception as e:
        out["error"] = (out["error"] or "") + f" tags: {type(e).__name__}: {e}"
    try:
        r = requests.post(f"{base_url.rstrip('/')}/api/show", json={"name": model}, timeout=timeout)
        if r.ok:
            det = r.json().get("details") or {}
            out.setdefault("quantization_level", det.get("quantization_level"))
            out["quantization_level"] = out["quantization_level"] or det.get("quantization_level")
            out["parameter_size"] = out["parameter_size"] or det.get("parameter_size")
            out["family"] = out["family"] or det.get("family")
    except Exception as e:
        out["error"] = (out["error"] or "") + f" show: {type(e).__name__}: {e}"
    return out


def corpus_metadata(df, text_column: str, id_column: Optional[str], source_path: str | Path,
                    n_raw: int, filter_description: str = "") -> dict:
    """Content-address the analyzed corpus and record exactly which rows were included.

    A row count alone cannot distinguish an 8-document corpus from a --max-essays 8 smoke run,
    and the corpus file itself is typically untracked, so the run must carry its own hashes.
    """
    ids = [str(x) for x in df[id_column].tolist()] if id_column else [str(i) for i in range(len(df))]
    texts = df[text_column].astype(str).tolist()
    row_hashes = sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts)
    lengths = [len(t.split()) for t in texts]
    p = Path(source_path)
    return {
        "source_path": str(p.resolve()) if p.exists() else str(p),
        "source_sha256": sha256_file(p),
        "source_bytes": p.stat().st_size if p.exists() else None,
        "n_raw_rows": n_raw,
        "n_analyzed": len(df),
        "n_excluded": n_raw - len(df),
        "filter_description": filter_description,
        "text_column": text_column,
        "id_column": id_column,
        "analyzed_ids": ids,
        "analyzed_ids_sha256": sha256_text("\n".join(ids)),
        "ids_unique": len(set(ids)) == len(ids),
        "content_sha256_order_independent": sha256_text("\n".join(row_hashes)),
        "word_count": {
            "total": int(sum(lengths)), "min": int(min(lengths)) if lengths else None,
            "max": int(max(lengths)) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else None,
        },
    }


def consolidation_settings(pipeline) -> dict:
    """The consolidation step's effective settings (Step 7 novelty evaluation), for the manifest.

    Exists because a run's consolidation configuration was historically only derivable by reading
    the code that launched it: replays that recorded their settings were auditable a month later,
    inline runs were not. Every field is read off the LIVE evaluator (which has already resolved
    policy/version/override precedence), not off the config the caller thinks it passed.

    Temperature is recorded as requested (None = defer to backend) alongside the backend's own
    default, rather than imputing one effective number -- an imputation whose justification lives
    only in code is exactly the failure this module exists to prevent.
    """
    ev = pipeline.novelty_evaluator
    cfg = pipeline.config
    return {
        "policy": ev.policy,
        "prompt_version": ev.prompt_version,
        "custom_prompt_override": (
            getattr(cfg, "novelty_evaluation_system_prompt", None) is not None
            or getattr(cfg, "novelty_evaluation_user_prompt", None) is not None
        ),
        "similarity_threshold": ev.similarity_threshold,
        "top_k_rag": ev.top_k_rag,
        "include_rejected_in_rag": ev.include_rejected_in_rag,
        "stage1_compare_accepted_only": ev.stage1_compare_accepted_only,
        "requested_temperature": ev.temperature,
        "backend_default_temperature": getattr(cfg.llm, "temperature", None),
        "study_context_present": ev.study_context is not None,
        "study_context_sha256": sha256_text(ev.study_context),
        "system_prompt_sha256": sha256_text(ev._system_prompt),
        "user_prompt_sha256": sha256_text(ev._user_prompt),
    }


def resolved_prompts(pipeline) -> dict:
    """Snapshot the prompt templates actually in force on the live objects.

    Reading templates off ``prompts.py`` is not sufficient: arm-B style overrides are injected at
    runtime (sometimes from files outside the repo), and ``generate_json`` appends a JSON-only
    instruction that never appears in the module source.
    """
    out: dict = {}

    def _snap(label, text, source):
        if text is None:
            return
        out[label] = {"source": source, "sha256": sha256_text(text), "text": text}

    try:
        from pygatos import prompts as P
        for name in ("GENERIC_SUMMARY_SYSTEM", "GENERIC_SUMMARY_PROMPT",
                     "INFORMATION_EXTRACTION_SYSTEM", "INFORMATION_EXTRACTION_PROMPT",
                     "THEME_SUGGESTION_SYSTEM", "THEME_SUGGESTION_PROMPT",
                     "THEME_ASSIGNMENT_SYSTEM", "THEME_ASSIGNMENT_PROMPT",
                     "CODE_APPLICATION_SYSTEM_V3", "CODE_APPLICATION_PROMPT_V3"):
            _snap(name, getattr(P, name, None), f"pygatos.prompts.{name}")
    except Exception as e:
        out["_prompts_module_error"] = str(e)

    # live objects (these carry any runtime overrides)
    for attr_path, label in [
        ("novelty_evaluator._system_prompt", "novelty_system_effective"),
        ("novelty_evaluator._user_prompt", "novelty_user_effective"),
        ("code_suggester.system_prompt", "code_suggestion_system_effective"),
        ("code_suggester.user_prompt", "code_suggestion_user_effective"),
    ]:
        obj = pipeline
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            _snap(label, obj if isinstance(obj, str) else None, f"live:{attr_path}")
        except Exception:
            continue

    out["_generate_json_suffix"] = {
        "text": "You must respond with valid JSON only. No additional text or explanation.",
        "note": "appended to every system prompt by OllamaBackend.generate_json; not visible in prompts.py",
    }
    return out

"""LLM-judged true-duplicate rate: a threshold-independent parsimony diagnostic.

Why this exists
---------------
The keep-unless-duplicate consolidation policy (``NoveltyConfig.policy``) buys its recall gain at
a known cost: the codebook grows, and on weak generator models genuinely duplicated codes get
through. A deployment that cannot measure recall (no ground truth) can still measure THIS.

Why not a cosine-threshold statistic: Stage 1 auto-rejects any candidate whose similarity to a
previously seen code exceeds the consolidation threshold tau, and cosine is symmetric, so no pair
of accepted codes can exceed tau -- "redundancy@t" is identically 0 for t >= tau by construction,
not by parsimony. This diagnostic instead asks an LLM judge whether each highly-similar PAIR of
accepted codes is a true duplicate, a legitimate broader/narrower sub-theme relation, or
distinct -- the same distinction a human coder makes, and one definition-embedding cosine cannot.

    dupe_rate = judged DUPLICATE pairs / judged pairs   (subtheme_rate, distinct_rate likewise)

Metric definition (ported verbatim from the validation harness, 2026-08-19; keep in sync with
``experiments/pygatos-live-eval/dupe_rate.py`` in the mars-algo-v2 repo -- the constants and the
judging prompt ARE the instrument, and validated reference points assume them):

- Candidate pairs: the top-K (default 20) most similar accepted-code pairs by
  name+definition-embedding cosine, floored at MIN_COS (default 0.60). K is fixed per codebook so
  large and small codebooks are judged with equal effort and comparable denominators; pairs below
  the floor are counted distinct without a call.
- Pairs are judged in batches (default 10) per LLM call.
- Judge rule: SINGLE judge (the validation lane's D17 rule). NOTE for anyone comparing against
  harness artifacts: the harness's stored ``dupe_rate`` field is a stricter both-judges-agree
  panel number, which understates the cost of the repair; the manuscript reports the single-judge
  rule, and that is what this module computes. Measured on 219 validation codebooks the mean rate
  was 0.074 (both-agree) vs 0.124 (single deepseek judge).

Judge choice: pass any configured pygatos LLM backend -- a local Ollama judge keeps confidential
corpora local (code names/definitions are derived from the corpus and inherit its
confidentiality); a cloud judge (e.g. OpenRouter) sends them off-machine. The judge model should
DIFFER from the generator model: a model auditing its own output shares its biases
(shared-bias validity). Reference points: on a qwen3:30b generator the keep-unless-duplicate
policy reached a 0.62 single-judge duplicate rate on one corpus, vs 0.00-0.12 on cleaner stacks.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np

from pygatos.core.codebook import Code
from pygatos.llm.base import BaseLLM

logger = logging.getLogger(__name__)

TOP_K = 20      # most-similar pairs judged per codebook
MIN_COS = 0.60  # below this, do not spend a call: not plausibly duplicative
BATCH = 10      # pairs per LLM call

DUPLICATE_JUDGE_SYSTEM = (
    "You are an expert qualitative researcher auditing a codebook for redundancy. "
    "For each pair of codes you must decide the relationship between them."
)

DUPLICATE_JUDGE_PROMPT = """For each numbered pair of qualitative codes below, classify the relationship:

- "DUPLICATE": the two codes express the SAME concept at the SAME level of specificity. A coder
  would not be able to say which one a given excerpt belongs to. One of them should be removed.
- "SUBTHEME": the codes are related but at DIFFERENT levels of generality (one is a specific case
  of the other), or they capture different facets. A qualitative researcher could legitimately
  keep both, because the specific one preserves a distinction the broader one loses.
- "DISTINCT": the codes are about materially different things.

Judge only the pair in front of you. Do not assume that high similarity implies duplication.

{pairs}

Return ONLY a JSON object of the form:
{{"verdicts": [{{"n": 1, "label": "DUPLICATE|SUBTHEME|DISTINCT"}}, ...]}}
with exactly one entry per numbered pair."""

_LABELS = ("DUPLICATE", "SUBTHEME", "DISTINCT")


def _as_dicts(codes: Sequence[Union[Code, dict]]) -> list[dict]:
    out = []
    for c in codes:
        if isinstance(c, dict):
            out.append({"name": c["name"], "definition": c.get("definition", "")})
        else:
            out.append({"name": c.name, "definition": c.definition})
    return out


def top_pairs(codes: list[dict], embedder, top_k: int = TOP_K,
              min_cos: float = MIN_COS) -> tuple[list[tuple[int, int, float]], int]:
    """The top-k most similar code pairs by name+definition embedding cosine, floored at min_cos.

    Returns (pairs, n_pairs_above_floor); each pair is (i, j, cosine)."""
    if len(codes) < 2:
        return [], 0
    emb = np.asarray(embedder.embed([f"{c['name']}: {c['definition']}" for c in codes]))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T
    iu = np.triu_indices(len(codes), k=1)
    vals = sim[iu]
    order = np.argsort(-vals)
    pairs = []
    for idx in order[:top_k]:
        if vals[idx] < min_cos:
            break
        pairs.append((int(iu[0][idx]), int(iu[1][idx]), float(vals[idx])))
    return pairs, int((vals >= min_cos).sum())


def judge_pairs(judge: BaseLLM, codes: list[dict], pairs: list[tuple[int, int, float]],
                batch: int = BATCH) -> list[Optional[str]]:
    """Labels aligned with ``pairs`` (None where the judge failed or answered off-scale)."""
    labels: list[Optional[str]] = [None] * len(pairs)
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        lines = []
        for n, (i, j, _s) in enumerate(chunk, 1):
            lines.append(
                f"PAIR {n}\n"
                f"  A. {codes[i]['name']}: {codes[i]['definition']}\n"
                f"  B. {codes[j]['name']}: {codes[j]['definition']}")
        try:
            out = judge.generate_json(
                prompt=DUPLICATE_JUDGE_PROMPT.format(pairs="\n\n".join(lines)),
                system=DUPLICATE_JUDGE_SYSTEM,
                temperature=0.0,
            )
            got = {int(v["n"]): str(v["label"]).strip().upper()
                   for v in (out or {}).get("verdicts", []) if "n" in v and "label" in v}
        except Exception as e:
            logger.warning(f"duplicate-rate judge batch failed "
                           f"({type(e).__name__}: {str(e)[:80]}); pairs left unlabelled")
            got = {}
        for n in range(1, len(chunk) + 1):
            lab = got.get(n)
            if lab in _LABELS:
                labels[start + n - 1] = lab
    return labels


def compute_duplicate_rate(
    codes: Sequence[Union[Code, dict]],
    judge: BaseLLM,
    embedder,
    top_k: int = TOP_K,
    min_cos: float = MIN_COS,
    batch: int = BATCH,
    generator_model: Optional[str] = None,
) -> dict:
    """Judge the top-k most similar accepted-code pairs and return duplicate/subtheme/distinct rates.

    Args:
        codes: Accepted codes -- ``Code`` objects or dicts with ``name``/``definition``.
        judge: LLM backend used as the judge. Should be a DIFFERENT model from the generator
            (shared-bias validity); pass ``generator_model`` to get a loud warning if not.
        embedder: An object with ``embed(list[str]) -> array`` (e.g. ``pygatos.core.Embedder``).
        top_k / min_cos / batch: The metric's instrument constants; change them only if you
            do not need comparability with the validated reference points.
        generator_model: Optional model name of the generator that produced the codebook.

    Returns a dict with rates under the single-judge rule plus per-pair detail
    (``pairs[].label`` holds the judge's verdict; UNJUDGED pairs are excluded from every rate).
    """
    if generator_model is not None and generator_model == judge.model_name:
        logger.warning(
            "duplicate-rate judge model equals the generator model "
            f"({judge.model_name!r}): a model auditing its own output shares its biases, "
            "so this rate is not a valid independent check.")
    code_dicts = _as_dicts(codes)
    pairs, n_above = top_pairs(code_dicts, embedder, top_k=top_k, min_cos=min_cos)
    base = {"n_codes": len(code_dicts), "n_pairs_above_floor": n_above,
            "judge_model": judge.model_name, "rule": "single-judge",
            "top_k": top_k, "min_cos": min_cos, "batch": batch,
            "generator_model": generator_model}
    if not pairs:
        return {**base, "n_pairs_judged": 0, "n_pairs_unjudged": 0,
                "dupe_rate": 0.0, "subtheme_rate": 0.0, "distinct_rate": 0.0,
                "n_dupe_pairs": 0, "dupe_pairs_per_code": 0.0, "pairs": []}
    labels = judge_pairs(judge, code_dicts, pairs, batch=batch)
    counts = {lab: 0 for lab in _LABELS}
    unjudged = 0
    detail = []
    for (i, j, s), lab in zip(pairs, labels):
        if lab is None:
            unjudged += 1
        else:
            counts[lab] += 1
        detail.append({"a": code_dicts[i]["name"], "b": code_dicts[j]["name"],
                       "cos": round(s, 4), "label": lab or "UNJUDGED"})
    judged = sum(counts.values())
    return {
        **base,
        "n_pairs_judged": judged,
        "n_pairs_unjudged": unjudged,
        "dupe_rate": round(counts["DUPLICATE"] / judged, 4) if judged else 0.0,
        "subtheme_rate": round(counts["SUBTHEME"] / judged, 4) if judged else 0.0,
        "distinct_rate": round(counts["DISTINCT"] / judged, 4) if judged else 0.0,
        "n_dupe_pairs": counts["DUPLICATE"],
        "dupe_pairs_per_code": round(counts["DUPLICATE"] / len(code_dicts), 4) if code_dicts else 0.0,
        "pairs": detail,
    }

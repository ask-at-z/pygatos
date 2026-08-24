# Changelog

## 0.2.0 — 2026-08-24

### Added
- `NoveltyConfig.policy` — first-class consolidation policy: `"reject-unless-distinct"`
  (published default, unchanged behavior) or `"keep-unless-duplicate"` (the validated
  recall repair; +0.25–0.27 ground-truth recall on two model stacks, codebook grows 3–8×).
  The repair prompts are a verbatim port of the validation harness's arm-B instrument;
  tests pin both policies' prompt text by SHA-256 so neither can drift silently.
  Also exposed as `pygatos run --novelty-policy`.
- `pygatos.diagnostics.compute_duplicate_rate` — LLM-judged true-duplicate rate
  (single-judge rule), the no-ground-truth deployment diagnostic for the repair's known
  duplication cost. Judge is caller-supplied (local Ollama supported for confidential
  corpora); instrument constants and judging prompt ported verbatim and SHA-pinned.
- Forward-path run provenance in the CLI: every run writes `manifest.json` (effective
  consolidation settings incl. prompt hashes, environment/git/Ollama/corpus metadata,
  resolved prompts) and `llm_calls.jsonl` via `LLMCallRecorder` (on by default,
  `--no-llm-log` to disable). The manifest is written at run start (`status:
  "in_progress"`) so killed runs still leave their settings on disk.
- `pygatos.provenance.consolidation_settings()` — reads the effective Step-7 settings off
  the live evaluator (policy, thresholds, comparison set, retrieval, temperature as
  requested + backend default, prompt SHA-256s); also embedded in
  `pipeline_state.json` when `collect_provenance=True`.

### Unchanged (deliberately)
- The default consolidation policy remains the published `"reject-unless-distinct"` and the
  stock V2 prompt text is byte-identical (now SHA-pinned). Flipping the default is a
  versioning decision to be made when the consolidation manuscript is public.

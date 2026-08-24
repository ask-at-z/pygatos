"""Tests for the first-class novelty policy (the validated consolidation repair).

The SHA-256 pins are load-bearing: every validated number in the consolidation manuscript was
produced by the exact prompt wording, and this project's history is full of silent prompt drift.
If a pin fails, the prompt text changed -- that is a new instrument, not a refactor. Do not
update a pin without a deliberate, documented decision.
"""

import hashlib

import pytest

from pygatos import prompts as P
from pygatos.config import GATOSConfig, NoveltyConfig
from pygatos.generation.novelty_evaluator import NoveltyEvaluator, NOVELTY_POLICIES


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Frozen 2026-08-19 from experiments/pygatos-live-eval/arm_prompts/armB_novelty.json
# (mars-algo-v2 repo), the artifact every validated recall number was produced with.
ARMB_SYSTEM_SHA256 = "09e38b988009cd047edd3411f429d8d3f4bd9563afca1cda5d03b3062b5be2c4"
ARMB_USER_SHA256 = "8be3c0e1f39323895203e37f18ce205a3d25f6dd83bda3fcbb29254b84b15655"

# The published stock policy (prompt V2), pinned 2026-08-19 as shipped at package SHA f114906.
# The MS1 manuscript pins pygatos for reproducibility; changing this text silently would make
# "reject-unless-distinct" mean something the paper never measured.
PINNED_V2_SYSTEM_SHA256 = "2aa74f6e1821b3df798471b63e5cf7075bad5711f2d55fd2e12d966ddea898a9"
PINNED_V2_USER_SHA256 = "813d839fef4160736ea2d226340ed98667ec73512cb492110d55e28c761d6dfa"


class TestPromptPins:
    def test_keep_unless_duplicate_system_is_verbatim_armb(self):
        assert _sha(P.NOVELTY_EVALUATION_SYSTEM_KEEP_UNLESS_DUPLICATE) == ARMB_SYSTEM_SHA256

    def test_keep_unless_duplicate_user_is_verbatim_armb(self):
        assert _sha(P.NOVELTY_EVALUATION_PROMPT_KEEP_UNLESS_DUPLICATE) == ARMB_USER_SHA256

    def test_stock_v2_system_unchanged(self):
        assert _sha(P.NOVELTY_EVALUATION_SYSTEM_V2) == PINNED_V2_SYSTEM_SHA256

    def test_stock_v2_user_unchanged(self):
        assert _sha(P.NOVELTY_EVALUATION_PROMPT_V2) == PINNED_V2_USER_SHA256

    def test_user_prompt_has_required_placeholders(self):
        text = P.NOVELTY_EVALUATION_PROMPT_KEEP_UNLESS_DUPLICATE
        formatted = text.format(code_name="X", code_definition="Y", existing_codes="Z")
        assert "X" in formatted and "Y" in formatted and "Z" in formatted


class TestPolicySelection:
    def _evaluator(self, **kw):
        # llm/embedder are never touched by __init__'s prompt selection
        return NoveltyEvaluator(llm=object(), embedder=object(), **kw)

    def test_default_policy_is_published_behavior(self):
        ev = self._evaluator(prompt_version=2)
        assert ev.policy == "reject-unless-distinct"
        assert ev._system_prompt == P.NOVELTY_EVALUATION_SYSTEM_V2
        assert ev._user_prompt == P.NOVELTY_EVALUATION_PROMPT_V2

    def test_keep_unless_duplicate_selects_armb_prompts(self):
        ev = self._evaluator(policy="keep-unless-duplicate", prompt_version=2)
        assert ev._system_prompt == P.NOVELTY_EVALUATION_SYSTEM_KEEP_UNLESS_DUPLICATE
        assert ev._user_prompt == P.NOVELTY_EVALUATION_PROMPT_KEEP_UNLESS_DUPLICATE

    def test_policy_beats_prompt_version_1_too(self):
        ev = self._evaluator(policy="keep-unless-duplicate", prompt_version=1)
        assert ev._system_prompt == P.NOVELTY_EVALUATION_SYSTEM_KEEP_UNLESS_DUPLICATE

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown novelty policy"):
            self._evaluator(policy="keep-everything")

    def test_policy_plus_custom_prompts_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._evaluator(policy="keep-unless-duplicate", system_prompt="custom")

    def test_custom_prompts_with_default_policy_still_allowed(self):
        ev = self._evaluator(system_prompt="custom sys", user_prompt="custom usr {code_name}{code_definition}{existing_codes}")
        assert ev._system_prompt == "custom sys"

    def test_policy_registry_matches_config_literal(self):
        assert set(NOVELTY_POLICIES) == {"reject-unless-distinct", "keep-unless-duplicate"}
        assert NoveltyConfig().policy == "reject-unless-distinct"


class TestPipelineWiring:
    def test_config_policy_reaches_evaluator(self):
        from pygatos.pipeline import GATOSPipeline
        config = GATOSConfig()
        config.novelty.policy = "keep-unless-duplicate"
        pipe = GATOSPipeline(config=config)
        ev = pipe.novelty_evaluator
        assert ev.policy == "keep-unless-duplicate"
        assert ev._system_prompt == P.NOVELTY_EVALUATION_SYSTEM_KEEP_UNLESS_DUPLICATE

    def test_config_policy_and_custom_prompts_raise(self):
        from pygatos.pipeline import GATOSPipeline
        config = GATOSConfig()
        config.novelty.policy = "keep-unless-duplicate"
        config.novelty_evaluation_system_prompt = "custom"
        pipe = GATOSPipeline(config=config)
        with pytest.raises(ValueError, match="mutually exclusive"):
            pipe.novelty_evaluator

    def test_consolidation_settings_capture(self):
        from pygatos.pipeline import GATOSPipeline
        from pygatos.provenance import consolidation_settings, sha256_text
        config = GATOSConfig()
        config.novelty.policy = "keep-unless-duplicate"
        config.novelty.temperature = 0.0
        config.study_context = "a study about widgets"
        pipe = GATOSPipeline(config=config)
        s = consolidation_settings(pipe)
        assert s["policy"] == "keep-unless-duplicate"
        assert s["similarity_threshold"] == 0.8
        assert s["requested_temperature"] == 0.0
        assert s["backend_default_temperature"] == config.llm.temperature
        assert s["study_context_present"] is True
        assert s["custom_prompt_override"] is False
        assert s["system_prompt_sha256"] == ARMB_SYSTEM_SHA256
        assert s["user_prompt_sha256"] == ARMB_USER_SHA256
        assert s["study_context_sha256"] == sha256_text("a study about widgets")

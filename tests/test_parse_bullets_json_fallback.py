"""_parse_bullets is the fallback for FAILED JSON parses, so it must not ingest JSON scaffolding.

Regression test for the leak observed 2026-09-01: DeepSeek responses that failed strict JSON
parsing were line-parsed, and lines like '"information_points": [' entered extraction rounds as
information points (98 across three rounds), clustering together downstream.
"""
from unittest.mock import MagicMock

from pygatos.core.summarizer import Summarizer


def _parser():
    return Summarizer(llm=MagicMock())


def test_json_scaffolding_dropped():
    malformed = '\n'.join([
        '{',
        '"information_points": [',
        '"Faculty may use AI tools for drafting.",',
        '"Students must disclose AI use in submissions."',
        '],',
        '}',
    ])
    points = _parser()._parse_bullets(malformed)
    assert points == [
        "Faculty may use AI tools for drafting.",
        "Students must disclose AI use in submissions.",
    ]


def test_code_fence_dropped():
    fenced = '```json\n"AI use requires instructor permission."\n```'
    assert _parser()._parse_bullets(fenced) == ["AI use requires instructor permission."]


def test_plain_bullets_unchanged():
    plain = '- first point\n* second point\n3. third point'
    assert _parser()._parse_bullets(plain) == ["first point", "second point", "third point"]

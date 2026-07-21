"""Tests for provenance serialization and save_provenance / load_extraction_results."""

import json

from pygatos.models import InformationPoint
from pygatos.core.summarizer import SummarizationResult
from pygatos.core.codebook import Code, Codebook
from pygatos.config import GATOSConfig
from pygatos.pipeline import GATOSPipeline


def test_information_point_roundtrip():
    p = InformationPoint(text="students fear AI grading",
                         source_text="<full essay>", chunk_index=2, chunk_text="...AI grading...")
    p2 = InformationPoint.from_dict(p.to_dict())
    assert (p2.text, p2.source_text, p2.chunk_index, p2.chunk_text) == \
           (p.text, p.source_text, p.chunk_index, p.chunk_text)


def test_summarization_result_roundtrip_preserves_lineage():
    sr = SummarizationResult(
        original_text="full text here",
        chunks=["chunk one", "chunk two"],
        generic_summaries=["sum1", "sum2"],
        information_points=["point a", "point b"],
        n_chunks=2,
        structured_points=[
            InformationPoint("point a", "full text here", 0, "chunk one"),
            InformationPoint("point b", "full text here", 1, "chunk two"),
        ],
    )
    sr2 = SummarizationResult.from_dict(sr.to_dict())
    assert sr2.chunks == sr.chunks
    assert sr2.generic_summaries == sr.generic_summaries
    assert sr2.information_points == sr.information_points
    assert len(sr2.structured_points) == 2
    assert sr2.structured_points[1].chunk_index == 1
    assert sr2.structured_points[1].chunk_text == "chunk two"


def test_summarization_result_can_drop_original_text():
    sr = SummarizationResult("secret full essay", ["c"], ["s"], ["p"], 1,
                             [InformationPoint("p", "secret full essay", 0, "c")])
    d = sr.to_dict(include_original_text=False)
    assert d["original_text"] is None
    assert "secret full essay" not in json.dumps({"original_text": d["original_text"]})
    # chunks/points still present so reuse still works
    assert SummarizationResult.from_dict(d).information_points == ["p"]


def _pipeline_with_provenance():
    """A pipeline whose .provenance is set as generate_codebook(collect_provenance=True) would."""
    pipe = GATOSPipeline(config=GATOSConfig())
    # two documents, three points; point 2 belongs to cluster 0 with point 0
    all_points = ["students fear AI grading", "AI helps drafting", "AI grading is unfair"]
    point_to_source = {0: "essay_1.txt", 1: "essay_2.txt", 2: "essay_1.txt"}
    point_structured = {
        0: InformationPoint(all_points[0], "<e1>", 0, "chunk of e1"),
        1: InformationPoint(all_points[1], "<e2>", 0, "chunk of e2"),
        2: InformationPoint(all_points[2], "<e1>", 1, "chunk2 of e1"),
    }
    extraction = {
        "essay_1.txt": SummarizationResult("<e1>", ["chunk of e1", "chunk2 of e1"], ["s1", "s2"],
                                           [all_points[0], all_points[2]], 2,
                                           [point_structured[0], point_structured[2]]),
        "essay_2.txt": SummarizationResult("<e2>", ["chunk of e2"], ["s"], [all_points[1]], 1,
                                           [point_structured[1]]),
    }
    pipe.provenance = {
        "extraction_by_id": extraction,
        "point_to_source": point_to_source,
        "point_structured": point_structured,
        "all_points": all_points,
        "labels": [0, 1, 0],
        "ids": ["essay_1.txt", "essay_2.txt"],
    }
    cb = Codebook()
    c0 = Code(name="AI grading fairness", definition="concerns re AI grading", source_cluster=0)
    c0.metadata = {"source_essays": ["essay_1.txt"], "source_point_indices": [0, 2], "n_source_points": 2}
    c1 = Code(name="AI as drafting aid", definition="AI assists writing", source_cluster=1)
    c1.metadata = {"source_essays": ["essay_2.txt"], "source_point_indices": [1], "n_source_points": 1}
    cb.add_code(c0, accepted=True)
    cb.add_code(c1, accepted=True)
    return pipe, cb


def test_save_provenance_writes_grounding_and_extraction(tmp_path):
    pipe, cb = _pipeline_with_provenance()
    paths = pipe.save_provenance(cb, tmp_path)

    grounding = json.loads(paths["code_grounding"].read_text())
    by_code = {g["code"]: g for g in grounding}
    # the code links back to the exact information points + essays that induced it
    g0 = by_code["AI grading fairness"]
    assert g0["source_essays"] == ["essay_1.txt"]
    assert g0["n_source_points"] == 2
    texts = [p["text"] for p in g0["source_information_points"]]
    assert texts == ["students fear AI grading", "AI grading is unfair"]
    assert g0["source_information_points"][1]["chunk_index"] == 1
    assert g0["source_information_points"][1]["essay_id"] == "essay_1.txt"

    # extraction round-trips into usable SummarizationResults (for point reuse in application)
    extraction = GATOSPipeline.load_extraction_results(paths["extraction"])
    assert set(extraction) == {"essay_1.txt", "essay_2.txt"}
    assert extraction["essay_1.txt"].structured_points[0].chunk_text == "chunk of e1"


def test_save_provenance_without_run_raises(tmp_path):
    pipe = GATOSPipeline(config=GATOSConfig())
    cb = Codebook()
    try:
        pipe.save_provenance(cb, tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "provenance" in str(e).lower()

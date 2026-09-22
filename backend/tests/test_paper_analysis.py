import unittest
from unittest.mock import patch

from app.agents.paper_analysis_agent import (
    EvidenceField,
    PaperAnalysisOutput,
    analyze_paper,
    rank_field_candidates,
    retrieve_analysis_evidence,
    validate_analysis_citations
)


def chunk(chunk_id, text, distance=0.2, page=1):
    return {
        "chunk_id": chunk_id,
        "paper_id": 1,
        "page_number": page,
        "text": text,
        "distance": distance
    }


def empty_output(**updates):
    values = {
        field: EvidenceField()
        for field in (
            "objective", "methodology", "dataset", "model",
            "evaluation_metrics", "main_results", "limitations",
            "future_work"
        )
    }
    values.update(updates)
    return PaperAnalysisOutput(**values)


class PaperAnalysisRetrievalTests(unittest.TestCase):
    def test_paper_level_objective_is_preferred(self):
        candidates = [
            chunk(
                "P1_C2",
                "A reader study compares resident editing times.",
                0.10,
                8
            ),
            chunk(
                "P1_C1",
                "Abstract: This paper proposes a vision-language model "
                "for multiple interpretation tasks.",
                0.16,
                1
            )
        ]
        ranked = rank_field_candidates("objective", candidates)
        self.assertEqual(ranked[0]["chunk_id"], "P1_C1")

    def test_core_methodology_is_preferred(self):
        candidates = [
            chunk("P1_C2", "Images were resized to 224x224.", 0.10, 4),
            chunk(
                "P1_C1",
                "Methods: Our framework uses image-text data and "
                "multi-task training.",
                0.17,
                2
            )
        ]
        ranked = rank_field_candidates("methodology", candidates)
        self.assertEqual(ranked[0]["chunk_id"], "P1_C1")

    def test_model_architecture_evidence_is_preferred(self):
        candidates = [
            chunk("P1_C2", "The system drafts reports.", 0.10, 7),
            chunk(
                "P1_C1",
                "Model architecture: a vision-language model combines "
                "an image encoder with a text language component.",
                0.17,
                3
            )
        ]
        ranked = rank_field_candidates("model", candidates)
        self.assertEqual(ranked[0]["chunk_id"], "P1_C1")

    def test_explicit_limitation_is_preferred_to_failure_example(self):
        candidates = [
            chunk("P1_C2", "One failure case missed an opacity.", 0.10, 9),
            chunk(
                "P1_C1",
                "Discussion and limitations: external validation and "
                "generalizability remain limited.",
                0.18,
                10
            )
        ]
        ranked = rank_field_candidates("limitations", candidates)
        self.assertEqual(ranked[0]["chunk_id"], "P1_C1")

    @patch("app.agents.paper_analysis_agent.search_chunks")
    def test_multi_query_retrieval_deduplicates_and_limits_candidates(
        self,
        search
    ):
        search.return_value = [
            chunk("P1_C1", "Abstract: This paper proposes a framework."),
            chunk("P1_C2", "Additional evidence.", 0.3, 2)
        ]
        results = retrieve_analysis_evidence(1, top_k_per_field=3)
        self.assertEqual(
            len({item["chunk_id"] for item in results}),
            len(results)
        )
        self.assertGreater(search.call_count, 8)
        self.assertIn("objective", results[0]["retrieved_for"])

    def test_citation_must_match_field_retrieval_pool(self):
        analysis = empty_output(
            objective=EvidenceField(
                value="Objective",
                evidence_text="Result-only evidence",
                page_number=4,
                chunk_id="P1_C4"
            ),
            main_results=EvidenceField(
                value="Result",
                evidence_text="Result-only evidence",
                page_number=4,
                chunk_id="P1_C4"
            )
        )
        retrieved = [{
            **chunk("P1_C4", "Main result", page=4),
            "retrieved_for": ["main_results"]
        }]
        validated = validate_analysis_citations(analysis, retrieved)
        self.assertFalse(validated["objective"]["citation_valid"])
        self.assertTrue(validated["main_results"]["citation_valid"])

    @patch("app.agents.paper_analysis_agent.generate_structured_answer")
    @patch("app.agents.paper_analysis_agent.retrieve_analysis_evidence")
    def test_no_invented_future_work(self, retrieve, generate):
        retrieve.return_value = [{
            **chunk("P1_C1", "No explicit future direction.", page=1),
            "retrieved_for": ["objective"]
        }]
        generate.return_value = empty_output(
            objective=EvidenceField(
                value="Supported objective",
                evidence_text="No explicit future direction.",
                page_number=1,
                chunk_id="P1_C1"
            )
        )
        result = analyze_paper(1)
        self.assertIsNone(result["analysis"]["future_work"]["value"])
        self.assertFalse(
            result["analysis"]["future_work"]["citation_valid"]
        )
        prompt = generate.call_args.kwargs["prompt"]
        self.assertIn("Otherwise return null", prompt)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.evidence_verification_agent import (
    SemanticDecision,
    SemanticVerificationOutput,
    build_verification_summary,
    collect_report_claims,
    extract_numbers,
    validate_claim_deterministically,
    verify_evidence
)
from app.database.database import Base, get_db
from app.database.models import Paper, PaperAnalysis
from app.main import app as main_app
from app.routes.verification import router


def reference(paper_id, field, value, page=2, chunk=None, valid=True):
    return {
        "paper_id": paper_id, "field": field, "value": value,
        "evidence_text": value, "page_number": page,
        "chunk_id": chunk or f"P{paper_id}_C4", "citation_valid": valid
    }


def claim(text, paper_ids, refs, expected="main_results", claim_type="paper_fact"):
    return {
        "claim_id": "CLM_001", "claim": text, "claim_type": claim_type,
        "source_agent": "agent4", "source_section": "major_findings",
        "paper_ids": paper_ids, "expected_field": expected,
        "evidence_refs": refs
    }


class EvidenceVerificationAgentTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "P1_main_results": reference(
                1, "main_results", "The model achieved F1 score 0.435."
            ),
            "P1_dataset": reference(
                1, "dataset", "A dataset of radiographs was used.", chunk="P1_C7"
            ),
            "P1_limitations": reference(
                1, "limitations", "No patient history was included.",
                chunk="P1_C8"
            ),
            "P2_methodology": reference(
                2, "methodology", "Contrastive learning was used.",
                chunk="P2_C4"
            ),
            "P3_methodology": reference(
                3, "methodology", "Contrastive learning was used.",
                chunk="P3_C4"
            )
        }

    def ref(self, key, **overrides):
        result = {"evidence_key": key, **self.registry[key]}
        result.update(overrides)
        return result

    def test_valid_numeric_claim_is_fully_supported(self):
        result = validate_claim_deterministically(claim(
            "The model achieved F1 score 0.435.", [1],
            [self.ref("P1_main_results")]
        ), self.registry)
        self.assertEqual(result["status"], "fully_supported")
        self.assertEqual(result["numeric_check"]["supported_numbers"], ["0.435"])

    def test_wrong_number_is_unsupported(self):
        result = validate_claim_deterministically(claim(
            "The model achieved F1 score 0.95.", [1],
            [self.ref("P1_main_results")]
        ), self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("numeric_mismatch", [i["type"] for i in result["issues"]])

    def test_wrong_chunk_is_detected(self):
        result = validate_claim_deterministically(claim(
            "The model achieved F1 score 0.435.", [1],
            [self.ref("P1_main_results", chunk_id="P1_C7")]
        ), self.registry)
        self.assertFalse(result["citation_check"]["chunk_valid"])
        self.assertIn("chunk_mismatch", [i["type"] for i in result["issues"]])

    def test_wrong_page_is_detected(self):
        result = validate_claim_deterministically(claim(
            "The model achieved F1 score 0.435.", [1],
            [self.ref("P1_main_results", page_number=4)]
        ), self.registry)
        self.assertFalse(result["citation_check"]["page_valid"])
        self.assertIn("page_mismatch", [i["type"] for i in result["issues"]])

    def test_wrong_field_is_partial(self):
        result = validate_claim_deterministically(claim(
            "The model uses a transformer.", [1],
            [self.ref("P1_dataset")], expected="model"
        ), self.registry)
        self.assertEqual(result["status"], "partially_supported")
        self.assertIn("field_mismatch", [i["type"] for i in result["issues"]])

    def test_multi_paper_claim_requires_every_paper(self):
        full = validate_claim_deterministically(claim(
            "Contrastive learning", [2, 3],
            [self.ref("P2_methodology"), self.ref("P3_methodology")],
            expected="methodology", claim_type="cross_paper_fact"
        ), self.registry)
        partial = validate_claim_deterministically(claim(
            "Contrastive learning", [2, 3],
            [self.ref("P2_methodology")], expected="methodology",
            claim_type="cross_paper_fact"
        ), self.registry)
        self.assertEqual(full["status"], "fully_supported")
        self.assertEqual(partial["status"], "partially_supported")
        self.assertIn(
            "missing_paper_support", [i["type"] for i in partial["issues"]]
        )

    def test_invalid_evidence_key_is_unsupported(self):
        result = validate_claim_deterministically(claim(
            "Unsupported claim", [99],
            [{"evidence_key": "P99_fake", "paper_id": 99}]
        ), self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn(
            "invalid_evidence_key", [i["type"] for i in result["issues"]]
        )

    def test_overstatement_is_unsupported(self):
        result = validate_claim_deterministically(claim(
            "The model is superior in all clinical settings.", [1],
            [self.ref("P1_main_results")]
        ), self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("overstatement", [i["type"] for i in result["issues"]])

    def test_candidate_gap_can_be_supported_as_inference(self):
        result = validate_claim_deterministically(claim(
            "Patient history integration warrants further investigation.", [1],
            [self.ref("P1_limitations")], expected="candidate_gap",
            claim_type="candidate_gap"
        ), self.registry)
        self.assertEqual(result["status"], "fully_supported")

    def test_generated_question_is_a_generated_inference(self):
        result = validate_claim_deterministically(claim(
            "How should patient history integration be investigated?", [1],
            [self.ref("P1_limitations")], expected="generated_inference",
            claim_type="generated_inference"
        ), self.registry)
        self.assertEqual(result["claim_type"], "generated_inference")
        self.assertNotEqual(result["status"], "unsupported")

    def test_report_wrong_citation_number_is_unsupported(self):
        report_claims = collect_report_claims({
            "major_findings": "Workflow improved by 81.2%. [Paper 1, p.2, P1_C4]"
        }, self.registry)
        report_claims[0]["claim_id"] = "CLM_001"
        result = validate_claim_deterministically(report_claims[0], self.registry)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("81.2%", result["numeric_check"]["unsupported_numbers"])

    def test_extract_numbers_preserves_units(self):
        self.assertEqual(
            extract_numbers("0.435, 81.2%, 377,110 and 2.7 billion"),
            ["0.435", "81.2%", "377,110", "2.7 billion"]
        )

    def test_summary_is_deterministic(self):
        results = (
            [{"status": "fully_supported"}] * 8
            + [{"status": "partially_supported"}]
            + [{"status": "unsupported"}]
        )
        self.assertEqual(build_verification_summary(results), {
            "total_claims": 10, "fully_supported": 8,
            "partially_supported": 1, "unsupported": 1,
            "support_ratio": 0.8
        })

    @patch(
        "app.agents.evidence_verification_agent.generate_structured_answer"
    )
    def test_semantic_verification_is_batched(self, generate):
        papers = [
            {"paper_id": 1, "analysis": {
                "objective": reference(1, "objective", "First objective."),
            }},
            {"paper_id": 2, "analysis": {
                "objective": reference(2, "objective", "Second objective."),
            }}
        ]
        generate.return_value = SemanticVerificationOutput(decisions=[
            SemanticDecision(claim_id="CLM_001", status="fully_supported"),
            SemanticDecision(claim_id="CLM_002", status="fully_supported")
        ])
        result = verify_evidence(
            papers, verify_agent3=False, verify_agent4=False
        )
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["summary"]["total_claims"], 2)

    @patch(
        "app.agents.evidence_verification_agent.generate_structured_answer"
    )
    def test_semantic_result_cannot_override_missing_paper_support(self, generate):
        papers = [{
            "paper_id": paper_id,
            "analysis": {
                "methodology": reference(
                    paper_id, "methodology", "Contrastive learning was used.",
                    chunk=f"P{paper_id}_C4"
                )
            }
        } for paper_id in (2, 3)]
        p2 = reference(
            2, "methodology", "Contrastive learning was used.",
            chunk="P2_C4"
        )
        agent3 = {
            "similarities": [{
                "statement": "Contrastive learning", "paper_ids": [2, 3],
                "evidence": [{"evidence_key": "P2_methodology", **p2}]
            }]
        }
        generate.return_value = SemanticVerificationOutput(decisions=[
            SemanticDecision(claim_id="CLM_001", status="fully_supported")
        ])
        result = verify_evidence(
            papers, agent3_output=agent3, verify_agent2=False,
            verify_agent3=True, verify_agent4=False
        )
        self.assertEqual(
            result["claim_results"][0]["status"], "partially_supported"
        )


class EvidenceVerificationRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        cls.Session = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)
        test_app = FastAPI()
        test_app.include_router(router)

        def override_db():
            with cls.Session() as db:
                yield db

        test_app.dependency_overrides[get_db] = override_db
        cls.client = TestClient(test_app)

    def setUp(self):
        with self.Session() as db:
            db.query(PaperAnalysis).delete()
            db.query(Paper).delete()
            db.add(Paper(id=1, title="One"))
            db.add(PaperAnalysis(
                paper_id=1,
                main_results=reference(
                    1, "main_results", "The model achieved F1 score 0.435."
                )
            ))
            db.commit()

    @patch(
        "app.agents.evidence_verification_agent.generate_structured_answer",
        side_effect=RuntimeError("quota exceeded")
    )
    def test_gemini_failure_returns_deterministic_http_200(self, generate):
        response = self.client.post("/verification/check", json={
            "paper_ids": [1], "verify_agent2": True,
            "verify_agent3": False, "verify_agent4": False,
            "verify_report": False
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["total_claims"], 1)
        self.assertIn(
            "Semantic verification unavailable",
            response.json()["warnings"][0]
        )

    def test_duplicate_paper_ids_are_rejected(self):
        response = self.client.post("/verification/check", json={
            "paper_ids": [1, 1]
        })
        self.assertEqual(response.status_code, 422)

    def test_verification_route_is_registered_in_openapi(self):
        path = main_app.openapi()["paths"]["/verification/check"]
        self.assertIn("post", path)
        self.assertEqual(path["post"]["tags"], ["Evidence Verification"])


if __name__ == "__main__":
    unittest.main()

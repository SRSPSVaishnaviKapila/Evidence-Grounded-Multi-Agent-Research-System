import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.cross_paper_agent import (
    ComparabilityAssessment,
    ComparisonCaution,
    ComparisonFinding,
    CrossPaperLLMOutput,
    PotentialConflict,
    build_dataset_analysis,
    build_metric_analysis,
    classify_method_type,
    compare_papers,
    create_evidence_registry,
    detect_missing_information,
    validate_findings
)
from app.database.database import Base, get_db
from app.database.models import Paper, PaperAnalysis
from app.routes.comparison import router


FIELDS = (
    "objective",
    "methodology",
    "dataset",
    "model",
    "evaluation_metrics",
    "main_results",
    "limitations",
    "future_work"
)


def analysis_field(paper_id, field_name):
    return {
        "value": f"Paper {paper_id} {field_name}",
        "evidence_text": f"Evidence for {field_name}",
        "page_number": paper_id,
        "chunk_id": f"P{paper_id}_C1",
        "citation_valid": True
    }


def agent_paper(paper_id):
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2020 + paper_id,
        "analysis": {
            field_name: analysis_field(
                paper_id,
                field_name
            )
            for field_name in FIELDS
        }
    }


class CrossPaperAgentTests(unittest.TestCase):
    def test_one_paper_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least two"
        ):
            compare_papers([agent_paper(1)])

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_result_shape_and_evidence_validation(
        self,
        generate
    ):
        generate.return_value = CrossPaperLLMOutput(
            similarities=[ComparisonFinding(
                statement="Shared objective",
                paper_ids=[1, 2],
                evidence_keys=[
                    "P1_objective",
                    "invented_key"
                ]
            )],
            differences=[ComparisonFinding(
                statement="Different models",
                paper_ids=[1, 2],
                evidence_keys=[
                    "P1_model",
                    "P2_model"
                ]
            )],
            potential_conflicts=[PotentialConflict(
                topic="Results",
                paper_ids=[1, 2],
                description="Potentially different finding",
                evidence_keys=["P1_main_results"]
            )],
            comparability_assessment=[ComparabilityAssessment(
                paper_ids=[1, 2],
                dataset_comparable=True,
                task_comparable=True,
                metric_comparable=True,
                evaluation_setup_comparable=True,
                direct_result_comparison_allowed=True,
                reason="Comparable setup"
            )]
        )

        result = compare_papers([
            agent_paper(1),
            agent_paper(2)
        ])

        for key in (
            "comparison_table",
            "similarities",
            "differences",
            "comparison_cautions",
            "comparability_assessment",
            "dataset_analysis",
            "metric_analysis",
            "method_trends",
            "common_limitations",
            "potential_conflicts",
            "missing_information",
            "evidence_coverage",
            "overall_observations"
        ):
            self.assertIn(key, result)

        self.assertEqual(result["similarities"], [])
        self.assertEqual(result["potential_conflicts"], [])
        self.assertEqual(len(result["differences"]), 1)

        self.assertEqual(
            result["comparison_table"][0]["dataset"],
            "Paper 1 dataset"
        )
        self.assertEqual(
            result["comparison_table"][0]["year"],
            2021
        )

    def test_evidence_completeness_statuses(self):
        papers = [agent_paper(1), agent_paper(2)]
        registry = create_evidence_registry(papers)
        findings = [
            ComparisonFinding(
                statement="Partial",
                paper_ids=[1, 2],
                evidence_keys=["P1_objective"],
                evidence_fields=["objective"]
            ),
            ComparisonFinding(
                statement="Full",
                paper_ids=[1, 2],
                evidence_keys=["P1_objective", "P2_objective"],
                evidence_fields=["objective"]
            ),
            ComparisonFinding(
                statement="None",
                paper_ids=[1, 2],
                evidence_keys=["invented"],
                evidence_fields=["objective"]
            )
        ]
        results = validate_findings(findings, registry)
        self.assertEqual(
            [item["evidence_status"] for item in results],
            ["partially_supported", "fully_supported", "unsupported"]
        )
        self.assertEqual(
            [item["evidence_valid"] for item in results],
            [False, True, False]
        )

    def test_cross_field_evidence_is_rejected(self):
        papers = [agent_paper(1), agent_paper(2)]
        papers[1]["analysis"]["objective"] = None
        registry = create_evidence_registry(papers)
        result = validate_findings([
            ComparisonFinding(
                statement="Objective comparison",
                paper_ids=[1, 2],
                evidence_keys=["P1_objective", "P2_main_results"],
                evidence_fields=["objective"]
            )
        ], registry)[0]
        self.assertEqual(result["evidence_status"], "partially_supported")
        self.assertEqual(
            [item["evidence_key"] for item in result["evidence"]],
            ["P1_objective"]
        )

    def test_missing_information_is_deterministic(self):
        papers = [agent_paper(1), agent_paper(2)]
        papers[1]["analysis"]["objective"] = None
        papers[1]["analysis"]["future_work"]["evidence_text"] = None
        registry = create_evidence_registry(papers)
        result = detect_missing_information(papers, registry)
        self.assertEqual(result, [{
            "paper_id": 2,
            "missing_fields": ["objective", "future_work"]
        }])

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_noncomparable_results_are_caution_not_conflict(self, generate):
        generate.return_value = CrossPaperLLMOutput(
            comparison_cautions=[ComparisonCaution(
                topic="Performance comparison",
                paper_ids=[1, 2],
                statement="Results are not directly comparable.",
                reason="Datasets and metrics differ.",
                evidence_keys=[
                    "P1_dataset", "P1_evaluation_metrics",
                    "P2_dataset", "P2_evaluation_metrics"
                ],
                evidence_fields=["dataset", "evaluation_metrics"]
            )],
            comparability_assessment=[ComparabilityAssessment(
                paper_ids=[1, 2],
                dataset_comparable=False,
                task_comparable=True,
                metric_comparable=False,
                evaluation_setup_comparable=False,
                direct_result_comparison_allowed=False,
                reason="Datasets and metrics differ."
            )],
            potential_conflicts=[PotentialConflict(
                topic="Performance",
                paper_ids=[1, 2],
                description="Different values",
                evidence_keys=["P1_main_results", "P2_main_results"]
            )]
        )
        result = compare_papers([agent_paper(1), agent_paper(2)])
        self.assertEqual(len(result["comparison_cautions"]), 1)
        self.assertEqual(result["potential_conflicts"], [])

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_comparable_conflict_requires_both_papers(self, generate):
        generate.return_value = CrossPaperLLMOutput(
            comparability_assessment=[ComparabilityAssessment(
                paper_ids=[1, 2],
                dataset_comparable=True,
                task_comparable=True,
                metric_comparable=True,
                evaluation_setup_comparable=True,
                direct_result_comparison_allowed=True,
                reason="Comparable designs"
            )],
            potential_conflicts=[PotentialConflict(
                topic="Finding",
                paper_ids=[1, 2],
                description="Different findings",
                conflict_type="genuine_finding_difference",
                evidence_keys=["P1_main_results", "P2_main_results"]
            )]
        )
        result = compare_papers([agent_paper(1), agent_paper(2)])
        conflict = result["potential_conflicts"][0]
        self.assertTrue(conflict["evidence_valid"])
        self.assertEqual(conflict["evidence_status"], "fully_supported")

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_missing_limitation_is_not_common(self, generate):
        papers = [agent_paper(1), agent_paper(2)]
        papers[1]["analysis"]["limitations"] = None
        generate.return_value = CrossPaperLLMOutput(
            common_limitations=[ComparisonFinding(
                statement="Shared limitation",
                paper_ids=[1, 2],
                evidence_keys=["P1_limitations"]
            )]
        )
        result = compare_papers(papers)
        self.assertEqual(result["common_limitations"], [])
        self.assertIn(
            "limitations",
            result["missing_information"][0]["missing_fields"]
        )

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_caution_infers_paper_ids_from_evidence(self, generate):
        generate.return_value = CrossPaperLLMOutput(
            comparison_cautions=[ComparisonCaution(
                topic="Direct Result Comparison",
                paper_ids=[],
                statement="Direct comparison is inappropriate.",
                reason="Datasets and metrics differ.",
                evidence_keys=[
                    "P1_dataset", "P1_evaluation_metrics",
                    "P2_dataset", "P2_evaluation_metrics"
                ],
                evidence_fields=["dataset", "evaluation_metrics"]
            )]
        )
        result = compare_papers([agent_paper(1), agent_paper(2)])
        caution = result["comparison_cautions"][0]
        self.assertEqual(caution["paper_ids"], [1, 2])
        self.assertTrue(caution["evidence_valid"])
        self.assertEqual(caution["evidence_status"], "fully_supported")

    def test_dataset_names_are_normalized(self):
        papers = [agent_paper(1)]
        papers[0]["analysis"]["dataset"]["value"] = (
            "The study used the MIMIC-CXR dataset, PadChest data, "
            "and the CheXpert test dataset."
        )
        registry = create_evidence_registry(papers)
        table = [{
            "paper_id": 1,
            "dataset": papers[0]["analysis"]["dataset"]["value"]
        }]
        result = build_dataset_analysis(table, registry)
        self.assertEqual(
            [item["name"] for item in result["datasets"]],
            ["MIMIC-CXR", "PadChest", "CheXpert"]
        )

    def test_metric_definition_does_not_create_false_metrics(self):
        papers = [agent_paper(1)]
        papers[0]["analysis"]["evaluation_metrics"]["value"] = (
            "F1 score, which is the harmonic average of precision "
            "and recall, and 95% bootstrap confidence intervals"
        )
        registry = create_evidence_registry(papers)
        table = [{
            "paper_id": 1,
            "evaluation_metrics": papers[0]["analysis"]
            ["evaluation_metrics"]["value"]
        }]
        result = build_metric_analysis(table, registry)
        self.assertEqual(
            [item["name"] for item in result["metrics"]],
            ["F1"]
        )

    def test_metric_aliases_are_normalized(self):
        papers = [agent_paper(1)]
        papers[0]["analysis"]["evaluation_metrics"]["value"] = (
            "Mean AUC, MCC, and F1 score"
        )
        registry = create_evidence_registry(papers)
        table = [{
            "paper_id": 1,
            "evaluation_metrics": papers[0]["analysis"]
            ["evaluation_metrics"]["value"]
        }]
        result = build_metric_analysis(table, registry)
        self.assertEqual(
            [item["name"] for item in result["metrics"]],
            ["AUROC", "MCC", "F1"]
        )

    def test_method_categories_are_short_and_deterministic(self):
        self.assertEqual(
            classify_method_type(
                "Supervised training",
                "DenseNet convolutional neural network"
            ),
            "Supervised CNN"
        )
        self.assertEqual(
            classify_method_type(
                "Self-supervised image-text learning",
                "Image encoder and text encoder"
            ),
            "Vision-language self-supervised learning"
        )

    def test_method_classification_priority(self):
        cases = (
            (
                "Supervised training",
                "DenseNet architecture with dense connections and batch normalization.",
                "Supervised CNN"
            ),
            (
                "The model is trained using self-supervised learning on "
                "chest X-rays and radiology reports.",
                "A Vision Transformer is used as the image encoder and a "
                "Transformer as the text encoder.",
                "Vision-language self-supervised learning"
            ),
            (
                "The image encoder and text encoder are jointly trained "
                "using contrastive image-text learning.",
                None,
                "Contrastive vision-language pretraining"
            ),
            (
                "A multimodal vision-language foundation model is trained "
                "for multiple medical image interpretation tasks.",
                None,
                "Vision-language foundation model"
            ),
            (
                "Image classification",
                "A Vision Transformer is used for image classification.",
                "Transformer-based classifier"
            )
        )
        for methodology, model, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_method_type(methodology, model),
                    expected
                )

    def test_contrastive_priority_over_self_supervised(self):
        self.assertEqual(
            classify_method_type(
                "Self-supervised contrastive image-text learning",
                "Image encoder and text encoder"
            ),
            "Contrastive vision-language pretraining"
        )

    def test_cnn_without_supervision_is_generic_classifier(self):
        self.assertEqual(
            classify_method_type(None, "A ResNet convolutional model"),
            "CNN classifier"
        )

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_missing_field_comparison_is_removed(self, generate):
        papers = [agent_paper(1), agent_paper(2)]
        papers[1]["analysis"]["objective"] = None
        papers[1]["analysis"]["limitations"] = None
        generate.return_value = CrossPaperLLMOutput(
            similarities=[ComparisonFinding(
                statement="Shared objective",
                paper_ids=[1, 2],
                evidence_keys=["P1_objective", "P2_objective"],
                evidence_fields=["objective"]
            )],
            differences=[ComparisonFinding(
                statement="Different limitations",
                paper_ids=[1, 2],
                evidence_keys=["P1_limitations", "P2_limitations"],
                evidence_fields=["limitations"]
            )]
        )
        result = compare_papers(papers)
        self.assertEqual(result["similarities"], [])
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["evidence_coverage"]["total_findings"], 0)
        missing = result["missing_information"][0]["missing_fields"]
        self.assertIn("objective", missing)
        self.assertIn("limitations", missing)

    @patch(
        "app.agents.cross_paper_agent."
        "generate_structured_answer"
    )
    def test_unsupported_finding_is_removed(self, generate):
        generate.return_value = CrossPaperLLMOutput(
            similarities=[ComparisonFinding(
                statement="Unsupported similarity",
                paper_ids=[1, 2],
                evidence_keys=["invented_key"]
            )]
        )
        result = compare_papers([agent_paper(1), agent_paper(2)])
        self.assertEqual(result["similarities"], [])
        self.assertEqual(result["evidence_coverage"], {
            "total_findings": 0,
            "fully_supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "coverage_ratio": 0.0
        })


class CrossPaperRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        cls.Session = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)

        app = FastAPI()
        app.include_router(router)

        def override_db():
            db = cls.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        cls.client = TestClient(app)

    def setUp(self):
        with self.Session() as db:
            db.query(PaperAnalysis).delete()
            db.query(Paper).delete()
            for paper_id in (1, 2, 3):
                db.add(Paper(
                    id=paper_id,
                    title=f"Paper {paper_id}"
                ))
            db.commit()

    def add_analysis(self, paper_id):
        with self.Session() as db:
            values = {
                field_name: analysis_field(
                    paper_id,
                    field_name
                )
                for field_name in FIELDS
            }
            db.add(PaperAnalysis(
                paper_id=paper_id,
                **values
            ))
            db.commit()

    def test_request_with_one_id_fails(self):
        response = self.client.post(
            "/comparison/",
            json={"paper_ids": [1]}
        )
        self.assertEqual(response.status_code, 422)

    def test_duplicate_ids_fail(self):
        response = self.client.post(
            "/comparison/",
            json={"paper_ids": [1, 1]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unique", response.json()["detail"])

    def test_missing_paper_returns_404(self):
        response = self.client.post(
            "/comparison/",
            json={"paper_ids": [1, 999]}
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("999", response.json()["detail"])

    def test_unanalyzed_paper_returns_400(self):
        self.add_analysis(1)
        response = self.client.post(
            "/comparison/",
            json={"paper_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("analyzed first", response.json()["detail"])

    @patch("app.routes.comparison.compare_papers")
    def test_two_analyzed_papers_return_result(
        self,
        compare
    ):
        self.add_analysis(1)
        self.add_analysis(2)
        compare.return_value = {
            "paper_ids": [1, 2],
            "comparison_table": [],
            "similarities": [],
            "differences": [],
            "comparison_cautions": [],
            "comparability_assessment": [],
            "dataset_analysis": {},
            "metric_analysis": {},
            "method_trends": [],
            "common_limitations": [],
            "potential_conflicts": [],
            "missing_information": [],
            "evidence_coverage": {},
            "overall_observations": []
        }

        response = self.client.post(
            "/comparison/",
            json={"paper_ids": [1, 2]}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_papers"], 2)
        for key in compare.return_value:
            self.assertIn(key, body)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.research_gap_report_agent import (
    CandidateGap,
    GapExtensionLLMOutput,
    RecurringLimitation,
    ReportClaim,
    ReportSection,
    ResearchGapLLMOutput,
    ResearchOpportunityLLM,
    ResearchQuestionLLM,
    ResearchReportLLMOutput,
    analyze_research_gaps,
    build_agent4_evidence_registry,
    build_evidence_diversity,
    build_gap_evidence_matrix,
    build_gap_traceability,
    build_recurring_limitations,
    calculate_gap_strength,
    deduplicate_underexplored_areas,
    deduplicate_research_opportunities,
    generate_research_report,
    generate_gap_extensions,
    normalize_gap_type,
    validate_report_claim,
    validate_candidate_gaps,
    validate_research_opportunities,
    validate_research_questions
)
from app.database.database import Base, get_db
from app.database.models import Paper, PaperAnalysis
from app.routes.gaps import router


FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations", "future_work"
)


def field(paper_id, name, value=None, valid=True):
    actual = value if value is not None else f"Paper {paper_id} {name}"
    return {
        "value": actual,
        "evidence_text": f"Evidence for {actual}",
        "page_number": paper_id,
        "chunk_id": f"P{paper_id}_C{name}",
        "citation_valid": valid
    }


def agent_paper(paper_id):
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2020 + paper_id,
        "analysis": {
            name: field(paper_id, name)
            for name in FIELDS
        }
    }


def empty_cross():
    return {
        "comparison_table": [], "similarities": [], "differences": [],
        "comparison_cautions": [], "comparability_assessment": [],
        "dataset_analysis": {}, "metric_analysis": {}, "method_trends": [],
        "common_limitations": [], "potential_conflicts": [],
        "missing_information": [], "evidence_issues": [],
        "overall_observations": []
    }


def report_output():
    section = ReportSection(
        claims=[ReportClaim(
            text="Within the analyzed papers, Paper 1 reports a limitation.",
            evidence_keys=["P1_limitations"],
            paper_ids=[1]
        )]
    )
    return ResearchReportLLMOutput(
        title="Evidence-grounded synthesis",
        executive_summary=section,
        research_landscape=section,
        methodological_evolution=section,
        dataset_and_evaluation_patterns=section,
        major_findings=section,
        recurring_limitations=section,
        candidate_research_gaps=section,
        unresolved_issues=section,
        conclusion=section
    )


class ResearchGapAgentTests(unittest.TestCase):
    def setUp(self):
        self.papers = [agent_paper(1), agent_paper(2)]
        self.registry = build_agent4_evidence_registry(
            self.papers, empty_cross()
        )

    def validated_gap(self, paper_ids=None, evidence_keys=None, title="Gap"):
        paper_ids = paper_ids or [1]
        evidence_keys = evidence_keys or ["P1_limitations"]
        return validate_candidate_gaps([CandidateGap(
            title=title,
            description="A validated issue within the selected papers.",
            supporting_paper_ids=paper_ids,
            evidence_keys=evidence_keys,
            confidence="high",
            reasoning="Validated evidence"
        )], self.registry)[0]

    def test_minimum_papers(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            analyze_research_gaps([self.papers[0]], empty_cross())

    def test_unknown_key_and_unsupported_gap_are_removed(self):
        gaps = [CandidateGap(
            title="Unsupported gap",
            description="Unsupported description",
            supporting_paper_ids=[1],
            evidence_keys=["P99_fake_limitation"],
            confidence="high",
            reasoning="Unknown evidence"
        )]
        self.assertEqual(validate_candidate_gaps(gaps, self.registry), [])

    def test_invalid_citation_is_not_full_support(self):
        papers = [agent_paper(1), agent_paper(2)]
        papers[1]["analysis"]["limitations"]["citation_valid"] = False
        registry = build_agent4_evidence_registry(papers, empty_cross())
        gaps = validate_candidate_gaps([CandidateGap(
            title="Validation limitation",
            description="The analyzed papers report validation constraints.",
            supporting_paper_ids=[1, 2],
            evidence_keys=["P1_limitations", "P2_limitations"],
            confidence="high",
            reasoning="Explicit limitations"
        )], registry)
        self.assertEqual(gaps[0]["evidence_status"], "partially_supported")
        self.assertFalse(gaps[0]["evidence_valid"])

    def test_repeated_explicit_limitations_support_high_confidence(self):
        gaps = validate_candidate_gaps([CandidateGap(
            title="Limited external validation",
            description="Both selected papers report limited validation.",
            supporting_paper_ids=[1, 2],
            evidence_keys=["P1_limitations", "P2_limitations"],
            confidence="high",
            reasoning="Repeated explicit limitations"
        )], self.registry)
        self.assertEqual(gaps[0]["confidence"], "high")
        self.assertEqual(gaps[0]["evidence_status"], "fully_supported")
        self.assertEqual(gaps[0]["support_scope"], "cross_paper")
        self.assertTrue(gaps[0]["cross_paper_support"])

    def test_single_paper_support_scope_and_confidence(self):
        gaps = validate_candidate_gaps([CandidateGap(
            title="Model scaling uncertainty",
            description="One selected paper identifies scaling as future work.",
            gap_type="scaling gap",
            supporting_paper_ids=[1],
            evidence_keys=["P1_limitations", "P1_future_work"],
            confidence="high",
            reasoning="Explicit limitation and future direction"
        )], self.registry)
        self.assertEqual(gaps[0]["gap_type"], "scaling_gap")
        self.assertEqual(gaps[0]["support_scope"], "single_paper")
        self.assertFalse(gaps[0]["cross_paper_support"])
        self.assertEqual(gaps[0]["confidence"], "medium")

    def test_gap_types_are_standardized(self):
        self.assertEqual(
            normalize_gap_type("missing patient clinical history"),
            "clinical_context_gap"
        )
        gap = CandidateGap(
            title="Clinical history",
            description="Selected papers provide limited context.",
            gap_type="missing patient clinical history",
            supporting_paper_ids=[1],
            evidence_keys=["P1_limitations"],
            confidence="medium",
            reasoning="Explicit limitation"
        )
        self.assertEqual(gap.gap_type, "clinical_context_gap")

    def test_research_question_is_linked_to_validated_gap(self):
        gap = self.validated_gap()
        questions = validate_research_questions([
            ResearchQuestionLLM(
                gap_id="GAP_001",
                question="How does the limitation affect evaluation?",
                question_type="evaluation",
                evidence_keys=["P1_limitations"]
            )
        ], [gap], self.registry)
        self.assertEqual(questions[0]["question_id"], "RQ_001")
        self.assertEqual(questions[0]["gap_id"], "GAP_001")
        self.assertEqual(questions[0]["evidence_status"], "fully_supported")

    def test_invalid_gap_id_discards_extensions(self):
        gap = self.validated_gap()
        questions = validate_research_questions([
            ResearchQuestionLLM(
                gap_id="GAP_999", question="What should be evaluated?",
                evidence_keys=["P1_limitations"]
            )
        ], [gap], self.registry)
        opportunities = validate_research_opportunities([
            ResearchOpportunityLLM(
                gap_id="GAP_999", title="Evaluation study",
                description="Evaluate the identified limitation.",
                opportunity_type="evaluation_framework",
                evidence_keys=["P1_limitations"]
            )
        ], [gap], self.registry)
        self.assertEqual(questions, [])
        self.assertEqual(opportunities, [])

    def test_extension_evidence_must_belong_to_gap(self):
        gap = self.validated_gap()
        questions = validate_research_questions([
            ResearchQuestionLLM(
                gap_id="GAP_001", question="What should be evaluated?",
                evidence_keys=["P2_limitations", "P99_fake"]
            )
        ], [gap], self.registry)
        self.assertEqual(questions, [])

    def test_gap_strength_uses_paper_breadth(self):
        single = self.validated_gap(
            evidence_keys=["P1_limitations", "P1_future_work"]
        )
        multiple = self.validated_gap(
            paper_ids=[1, 2],
            evidence_keys=["P1_limitations", "P2_limitations"],
            title="Repeated validation limitation"
        )
        self.assertEqual(single["gap_strength"], "moderate")
        self.assertEqual(multiple["gap_strength"], "strong")

    def test_evidence_diversity_counts_unique_papers(self):
        gap = self.validated_gap(
            evidence_keys=["P1_limitations", "P1_future_work"]
        )
        diversity = build_evidence_diversity(gap["evidence"])
        self.assertEqual(diversity["paper_count"], 1)
        self.assertEqual(diversity["evidence_count"], 2)
        self.assertEqual(calculate_gap_strength(gap["evidence"]), "moderate")

    def test_gap_matrix_contains_every_selected_paper(self):
        gap = self.validated_gap()
        matrix = build_gap_evidence_matrix([gap], [1, 2, 3, 4])
        support = {
            row["paper_id"]: row["support"]
            for row in matrix[0]["paper_support"]
        }
        self.assertEqual(support, {1: True, 2: False, 3: False, 4: False})

    def test_traceability_records_explicit_sources_and_removed_keys(self):
        evidence = [
            {"evidence_key": "P1_limitations", **self.registry["P1_limitations"]},
            {"evidence_key": "P1_future_work", **self.registry["P1_future_work"]}
        ]
        traceability = build_gap_traceability(
            evidence,
            ["P1_limitations", "P1_future_work", "P99_fake"]
        )
        self.assertTrue(traceability["explicit_limitation_support"])
        self.assertTrue(traceability["explicit_future_work_support"])
        self.assertEqual(traceability["unsupported_evidence_removed"], 1)

    def test_future_work_opportunity_has_deterministic_basis(self):
        gap = self.validated_gap(evidence_keys=["P1_future_work"])
        opportunities = validate_research_opportunities([
            ResearchOpportunityLLM(
                gap_id="GAP_001", title="Controlled follow-up study",
                description="Evaluate the stated future direction.",
                opportunity_type="evaluation_framework",
                generation_basis="other label",
                evidence_keys=["P1_future_work"]
            )
        ], [gap], self.registry)
        self.assertEqual(
            opportunities[0]["generation_basis"], "explicit_future_work"
        )
        self.assertEqual(opportunities[0]["confidence"], gap["confidence"])

    def test_duplicate_questions_for_same_gap_are_removed(self):
        gap = self.validated_gap(
            paper_ids=[1, 2],
            evidence_keys=["P1_limitations", "P2_limitations"]
        )
        questions = validate_research_questions([
            ResearchQuestionLLM(
                gap_id="GAP_001",
                question="How does this limitation affect model evaluation?",
                evidence_keys=["P1_limitations", "P2_limitations"]
            ),
            ResearchQuestionLLM(
                gap_id="GAP_001",
                question="How does this limitation affect the model evaluation?",
                evidence_keys=["P1_limitations", "P2_limitations"]
            )
        ], [gap], self.registry)
        self.assertEqual(len(questions), 1)

    def test_opportunity_does_not_duplicate_underexplored_area(self):
        opportunities = [{
            "opportunity_id": "OPP_001",
            "title": "Clinical context integration",
            "description": "Clinical context integration",
            "gap_id": "GAP_001"
        }, {
            "opportunity_id": "OPP_002",
            "title": "Controlled multimodal evaluation",
            "description": "Evaluate models using clinical context.",
            "gap_id": "GAP_001"
        }]
        result = deduplicate_research_opportunities(
            opportunities,
            [{"area": "  Clinical Context Integration  "}]
        )
        self.assertEqual(
            [item["opportunity_id"] for item in result], ["OPP_002"]
        )

    @patch(
        "app.agents.research_gap_report_agent.generate_structured_answer"
    )
    def test_gap_extensions_use_one_batched_llm_call(self, generate):
        first = self.validated_gap(title="Evaluation limitation")
        second = self.validated_gap(
            paper_ids=[2], evidence_keys=["P2_limitations"],
            title="Dataset limitation"
        )
        second["gap_id"] = "GAP_002"
        generate.return_value = GapExtensionLLMOutput(
            research_questions=[
                ResearchQuestionLLM(
                    gap_id="GAP_001", question="How should this be evaluated?",
                    evidence_keys=["P1_limitations"]
                ),
                ResearchQuestionLLM(
                    gap_id="GAP_002", question="Which data address this issue?",
                    question_type="dataset",
                    evidence_keys=["P2_limitations"]
                )
            ],
            research_opportunities=[]
        )
        questions, opportunities = generate_gap_extensions(
            [first, second], self.registry
        )
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(questions), 2)
        self.assertEqual(opportunities, [])

    def test_recurring_limitation_requires_two_papers(self):
        single = build_recurring_limitations([RecurringLimitation(
            topic="Single limitation",
            paper_ids=[1],
            description="Only one paper",
            evidence_keys=["P1_limitations"]
        )], self.registry)
        repeated = build_recurring_limitations([RecurringLimitation(
            topic="Repeated limitation",
            paper_ids=[1, 2],
            description="Two papers",
            evidence_keys=["P1_limitations", "P2_limitations"]
        )], self.registry)
        self.assertEqual(single, [])
        self.assertTrue(repeated[0]["evidence_valid"])

    @patch(
        "app.agents.research_gap_report_agent."
        "generate_structured_answer"
    )
    def test_missing_future_work_does_not_create_gap(self, generate):
        self.papers[1]["analysis"]["future_work"] = {
            "value": None, "evidence_text": None, "page_number": None,
            "chunk_id": None, "citation_valid": False
        }
        generate.side_effect = [
            ResearchGapLLMOutput(candidate_gaps=[CandidateGap(
                title="Missing future work",
                description="Future work was not extracted.",
                supporting_paper_ids=[2],
                evidence_keys=["P2_future_work"],
                confidence="low",
                reasoning="Missing value"
            )]),
            report_output()
        ]
        result = analyze_research_gaps(self.papers, empty_cross())
        self.assertEqual(result["candidate_gaps"], [])

    @patch(
        "app.agents.research_gap_report_agent."
        "generate_structured_answer",
        return_value=report_output()
    )
    def test_report_citations_are_resolved_from_registry(self, generate):
        report = generate_research_report(
            empty_cross(), {"candidate_gaps": []}, self.registry
        )
        self.assertIn(
            "[Paper 1, p.1, P1_Climitations]",
            report["executive_summary"]
        )
        self.assertNotIn("P99", str(report))

    def test_claim_level_citations_are_kept_separate(self):
        first = validate_report_claim(ReportClaim(
            text="The selected Paper 1 reports a limitation.",
            evidence_keys=["P1_limitations"],
            paper_ids=[1]
        ), self.registry)
        second = validate_report_claim(ReportClaim(
            text="The selected Paper 2 reports a limitation.",
            evidence_keys=["P2_limitations"],
            paper_ids=[2]
        ), self.registry)
        self.assertIn("P1_Climitations", first)
        self.assertNotIn("P2_Climitations", first)
        self.assertIn("P2_Climitations", second)
        self.assertNotIn("P1_Climitations", second)

    def test_invalid_claim_key_is_removed(self):
        claim = validate_report_claim(ReportClaim(
            text="Unsupported factual claim.",
            evidence_keys=["P99_fake_result"],
            paper_ids=[99]
        ), self.registry)
        self.assertIsNone(claim)

    def test_numeric_claim_requires_exact_value(self):
        self.registry["P1_main_results"]["value"] = (
            "The model achieved an F1 score of 0.435."
        )
        self.registry["P1_main_results"]["evidence_text"] = (
            "F1 score was 0.435 on the test set."
        )
        matched = validate_report_claim(ReportClaim(
            text="The selected model achieved an F1 score of 0.435.",
            evidence_keys=["P1_main_results"],
            paper_ids=[1]
        ), self.registry)
        mismatched = validate_report_claim(ReportClaim(
            text="The selected model achieved 95.2%.",
            evidence_keys=["P1_main_results"],
            paper_ids=[1]
        ), self.registry)
        self.assertIsNotNone(matched)
        self.assertIsNone(mismatched)

    def test_underexplored_area_deduplication(self):
        evidence = [{
            "evidence_key": "P1_limitations",
            **self.registry["P1_limitations"]
        }]
        common = {
            "paper_ids": [1], "confidence": "medium",
            "evidence": evidence, "evidence_valid": True,
            "evidence_status": "fully_supported"
        }
        result = deduplicate_underexplored_areas([
            {"area": "Integration of patient history",
             "basis": "Explicit limitation.", **common},
            {"area": "Clinical history integration in CXR models",
             "basis": "Same explicit evidence.", **common}
        ])
        self.assertEqual(len(result), 1)

    @patch(
        "app.agents.research_gap_report_agent."
        "generate_structured_answer",
        return_value=report_output()
    )
    def test_report_prompt_requires_conservative_scope(self, generate):
        generate_research_report(
            empty_cross(), {"candidate_gaps": []}, self.registry
        )
        prompt = generate.call_args.kwargs["prompt"]
        self.assertIn("within the analyzed papers", prompt)
        self.assertIn("the selected studies", prompt)
        self.assertIn("the reviewed paper set", prompt)

    @patch(
        "app.agents.research_gap_report_agent."
        "generate_structured_answer"
    )
    def test_end_to_end_filters_hallucinated_gap(self, generate):
        generate.side_effect = [
            ResearchGapLLMOutput(candidate_gaps=[CandidateGap(
                title="No research exists",
                description="No work has ever examined this topic.",
                supporting_paper_ids=[1, 2],
                evidence_keys=["P99_fake"],
                confidence="high",
                reasoning="Absence"
            )]),
            report_output()
        ]
        result = analyze_research_gaps(self.papers, empty_cross())
        self.assertEqual(result["candidate_gaps"], [])
        self.assertEqual(result["research_questions"], [])
        self.assertEqual(result["research_opportunities"], [])
        self.assertEqual(result["gap_evidence_matrix"], [])
        self.assertEqual(result["evidence_coverage"]["total_gaps"], 0)


class ResearchGapRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        cls.Session = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)
        app = FastAPI()
        app.include_router(router)

        def override_db():
            with cls.Session() as db:
                yield db

        app.dependency_overrides[get_db] = override_db
        cls.client = TestClient(app)

    def setUp(self):
        with self.Session() as db:
            db.query(PaperAnalysis).delete()
            db.query(Paper).delete()
            db.add_all([Paper(id=1, title="One"), Paper(id=2, title="Two")])
            db.commit()

    def test_one_paper_request_fails(self):
        response = self.client.post("/gaps/analyze", json={"paper_ids": [1]})
        self.assertEqual(response.status_code, 422)

    def test_missing_analysis_returns_clear_error(self):
        response = self.client.post(
            "/gaps/analyze", json={"paper_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("analyzed first", response.json()["detail"])

    @patch("app.routes.gaps.analyze_research_gaps")
    @patch("app.routes.gaps.compare_papers")
    def test_success_response_preserves_agent4_shape(self, compare, analyze):
        with self.Session() as db:
            for paper_id in (1, 2):
                db.add(PaperAnalysis(
                    paper_id=paper_id,
                    **{
                        name: field(paper_id, name)
                        for name in FIELDS
                    }
                ))
            db.commit()
        compare.return_value = empty_cross()
        analyze.return_value = {
            "paper_ids": [1, 2], "candidate_gaps": [],
            "recurring_limitations": [], "underexplored_areas": [],
            "unresolved_conflicts": [], "future_direction_signals": [],
            "gap_evidence_summary": {}, "report": {},
            "evidence_coverage": {}
        }
        response = self.client.post(
            "/gaps/analyze", json={"paper_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "Research gap analysis completed successfully"
        )

    @patch("app.routes.gaps.compare_papers")
    def test_gemini_quota_error_is_not_hidden_as_500(self, compare):
        with self.Session() as db:
            for paper_id in (1, 2):
                db.add(PaperAnalysis(
                    paper_id=paper_id,
                    **{name: field(paper_id, name) for name in FIELDS}
                ))
            db.commit()
        compare.side_effect = genai_errors.ClientError(
            429,
            {"error": {"status": "RESOURCE_EXHAUSTED"}},
            None
        )
        response = self.client.post(
            "/gaps/analyze", json={"paper_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 429)
        self.assertIn("quota exceeded", response.json()["detail"])

    @patch("app.routes.gaps.compare_papers")
    def test_gemini_connection_error_is_not_hidden_as_500(self, compare):
        with self.Session() as db:
            for paper_id in (1, 2):
                db.add(PaperAnalysis(
                    paper_id=paper_id,
                    **{name: field(paper_id, name) for name in FIELDS}
                ))
            db.commit()
        compare.side_effect = httpx.ConnectError("connection refused")
        response = self.client.post(
            "/gaps/analyze", json={"paper_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily unavailable", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

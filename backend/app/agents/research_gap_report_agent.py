import json
import re
from collections import Counter
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.llm_service import generate_structured_answer


Confidence = Literal["high", "medium", "low"]
GapType = Literal[
    "dataset_gap", "methodological_gap", "evaluation_gap",
    "generalization_gap", "clinical_validation_gap", "modality_gap",
    "clinical_context_gap", "annotation_gap", "benchmark_gap",
    "task_coverage_gap", "robustness_gap", "interpretability_gap",
    "deployment_gap", "efficiency_gap", "scaling_gap",
    "architecture_gap", "other"
]
SourceType = Literal[
    "explicit_future_work", "explicit_limitation",
    "cross_paper_inference"
]
QuestionType = Literal[
    "comparative", "performance", "generalization", "evaluation",
    "dataset", "multimodal", "clinical_validation", "scaling",
    "efficiency", "robustness", "interpretability", "deployment", "other"
]
OpportunityType = Literal[
    "model_development", "dataset_expansion", "multimodal_integration",
    "evaluation_framework", "external_validation", "clinical_study",
    "benchmark_development", "scaling_study", "robustness_study",
    "efficiency_optimization", "interpretability_study",
    "deployment_study", "other"
]
GenerationBasis = Literal[
    "explicit_future_work", "explicit_limitation",
    "cross_paper_inference", "conflict_resolution"
]
ANALYSIS_FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations", "future_work"
)


class CandidateGap(BaseModel):
    gap_id: Optional[str] = None
    title: str
    description: str
    gap_type: GapType = "other"
    supporting_paper_ids: list[int] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    reasoning: str

    @field_validator("gap_type", mode="before")
    @classmethod
    def standardize_gap_type(cls, value):
        return normalize_gap_type(value)


class RecurringLimitation(BaseModel):
    topic: str
    paper_ids: list[int] = Field(default_factory=list)
    description: str
    evidence_keys: list[str] = Field(default_factory=list)


class UnderexploredArea(BaseModel):
    area: str
    basis: str
    paper_ids: list[int] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"


class UnresolvedConflict(BaseModel):
    topic: str
    paper_ids: list[int] = Field(default_factory=list)
    description: str
    possible_explanations: list[str] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)


class FutureDirectionSignal(BaseModel):
    direction: str
    source_type: SourceType
    paper_ids: list[int] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)


class ResearchGapLLMOutput(BaseModel):
    candidate_gaps: list[CandidateGap] = Field(default_factory=list)
    recurring_limitations: list[RecurringLimitation] = Field(default_factory=list)
    underexplored_areas: list[UnderexploredArea] = Field(default_factory=list)
    unresolved_conflicts: list[UnresolvedConflict] = Field(default_factory=list)
    future_direction_signals: list[FutureDirectionSignal] = Field(
        default_factory=list
    )


class ResearchQuestionLLM(BaseModel):
    gap_id: str
    question: str
    question_type: QuestionType = "other"
    generation_basis: GenerationBasis = "cross_paper_inference"
    evidence_keys: list[str] = Field(default_factory=list)

    @field_validator("question_type", mode="before")
    @classmethod
    def standardize_question_type(cls, value):
        return normalize_choice(value, QUESTION_TYPES)

    @field_validator("generation_basis", mode="before")
    @classmethod
    def standardize_generation_basis(cls, value):
        return normalize_choice(
            value, GENERATION_BASES, "cross_paper_inference"
        )


class ResearchOpportunityLLM(BaseModel):
    gap_id: str
    title: str
    description: str
    opportunity_type: OpportunityType = "other"
    generation_basis: GenerationBasis = "cross_paper_inference"
    evidence_keys: list[str] = Field(default_factory=list)

    @field_validator("opportunity_type", mode="before")
    @classmethod
    def standardize_opportunity_type(cls, value):
        return normalize_choice(value, OPPORTUNITY_TYPES)

    @field_validator("generation_basis", mode="before")
    @classmethod
    def standardize_generation_basis(cls, value):
        return normalize_choice(
            value, GENERATION_BASES, "cross_paper_inference"
        )


class GapExtensionLLMOutput(BaseModel):
    research_questions: list[ResearchQuestionLLM] = Field(default_factory=list)
    research_opportunities: list[ResearchOpportunityLLM] = Field(
        default_factory=list
    )


class ReportClaim(BaseModel):
    text: str
    evidence_keys: list[str] = Field(default_factory=list)
    paper_ids: list[int] = Field(default_factory=list)


class ReportSection(BaseModel):
    claims: list[ReportClaim] = Field(default_factory=list)


class ResearchReportLLMOutput(BaseModel):
    title: str
    executive_summary: ReportSection
    research_landscape: ReportSection
    methodological_evolution: ReportSection
    dataset_and_evaluation_patterns: ReportSection
    major_findings: ReportSection
    recurring_limitations: ReportSection
    candidate_research_gaps: ReportSection
    research_questions: ReportSection = Field(default_factory=ReportSection)
    research_opportunities: ReportSection = Field(default_factory=ReportSection)
    unresolved_issues: ReportSection
    conclusion: ReportSection


GAP_TYPES = {
    "dataset_gap", "methodological_gap", "evaluation_gap",
    "generalization_gap", "clinical_validation_gap", "modality_gap",
    "clinical_context_gap", "annotation_gap", "benchmark_gap",
    "task_coverage_gap", "robustness_gap", "interpretability_gap",
    "deployment_gap", "efficiency_gap", "scaling_gap",
    "architecture_gap", "other"
}
QUESTION_TYPES = {
    "comparative", "performance", "generalization", "evaluation",
    "dataset", "multimodal", "clinical_validation", "scaling",
    "efficiency", "robustness", "interpretability", "deployment", "other"
}
OPPORTUNITY_TYPES = {
    "model_development", "dataset_expansion", "multimodal_integration",
    "evaluation_framework", "external_validation", "clinical_study",
    "benchmark_development", "scaling_study", "robustness_study",
    "efficiency_optimization", "interpretability_study",
    "deployment_study", "other"
}
GENERATION_BASES = {
    "explicit_future_work", "explicit_limitation",
    "cross_paper_inference", "conflict_resolution"
}


def normalize_choice(
    value: Any, allowed: set[str], fallback: str = "other"
) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value or "").casefold()
    ).strip("_")
    return normalized if normalized in allowed else fallback


def normalize_gap_type(value: Any) -> str:
    """Map free-form model labels to the stable public taxonomy."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    if normalized in GAP_TYPES:
        return normalized
    rules = (
        (("patient history", "clinical history", "clinical context"),
         "clinical_context_gap"),
        (("scale", "scaling"), "scaling_gap"),
        (("external validation", "clinical validation"),
         "clinical_validation_gap"),
        (("generaliz",), "generalization_gap"),
        (("dataset", "data scope", "population diversity"), "dataset_gap"),
        (("modality", "image view", "lateral view"), "modality_gap"),
        (("metric", "evaluation", "standardization"), "evaluation_gap"),
        (("benchmark",), "benchmark_gap"),
        (("annotat", "label"), "annotation_gap"),
        (("robust",), "robustness_gap"),
        (("interpret", "explain"), "interpretability_gap"),
        (("deploy",), "deployment_gap"),
        (("efficien", "memory", "latency"), "efficiency_gap"),
        (("architecture", "architectural"), "architecture_gap"),
        (("task coverage", "task scope", "multi task"), "task_coverage_gap"),
        (("method", "training"), "methodological_gap")
    )
    text = str(value or "").casefold()
    for signals, category in rules:
        if any(signal in text for signal in signals):
            return category
    return "other"


def _value(item: Any, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _has_text(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _valid_reference(reference: dict) -> bool:
    return (
        bool(reference.get("citation_valid"))
        and _has_text(reference.get("value"))
        and _has_text(reference.get("evidence_text"))
        and reference.get("page_number") is not None
        and _has_text(reference.get("chunk_id"))
    )


def build_agent4_evidence_registry(
    papers: list[Any],
    cross_paper_output: dict
) -> dict[str, dict]:
    """Build trusted evidence solely from Agents 2 and 3."""

    registry = {}
    for paper in papers:
        paper_id = int(_value(paper, "paper_id"))
        analysis = _value(paper, "analysis") or {}
        for field_name in ANALYSIS_FIELDS:
            field = _value(analysis, field_name) or {}
            support_type = {
                "limitations": "explicit_limitation",
                "future_work": "explicit_future_work",
                "dataset": "dataset_pattern",
                "methodology": "method_pattern",
                "model": "method_pattern"
            }.get(field_name, "other")
            registry[f"P{paper_id}_{field_name}"] = {
                "paper_id": paper_id,
                "field": field_name,
                "value": _value(field, "value"),
                "evidence_text": _value(field, "evidence_text"),
                "page_number": _value(field, "page_number"),
                "chunk_id": _value(field, "chunk_id"),
                "citation_valid": bool(_value(field, "citation_valid")),
                "support_types": [support_type]
            }

    source_mappings = {
        "differences": "cross_paper_difference",
        "comparison_cautions": "comparison_caution",
        "potential_conflicts": "potential_conflict"
    }
    for collection_name, support_type in source_mappings.items():
        for finding in cross_paper_output.get(collection_name, []):
            for evidence in finding.get("evidence", []):
                key = evidence.get("evidence_key")
                if key in registry:
                    types = registry[key]["support_types"]
                    if support_type not in types:
                        types.append(support_type)
    return registry


def build_gap_context(
    papers: list[Any],
    cross_paper_output: dict,
    registry: dict[str, dict]
) -> str:
    """Exclude pipeline-quality fields from scientific gap context."""

    allowed_cross_fields = (
        "comparison_table", "similarities", "differences",
        "comparison_cautions", "comparability_assessment",
        "dataset_analysis", "metric_analysis", "method_trends",
        "common_limitations", "potential_conflicts",
        "overall_observations"
    )
    context = {
        "paper_ids": [_value(paper, "paper_id") for paper in papers],
        "cross_paper_analysis": {
            key: cross_paper_output.get(key)
            for key in allowed_cross_fields
        },
        "evidence_registry": registry
    }
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def validate_gap_evidence(
    evidence_keys: list[str],
    paper_ids: list[int],
    registry: dict[str, dict],
    allowed_fields: Optional[set[str]] = None
) -> list[dict]:
    evidence, seen = [], set()
    allowed_papers = set(paper_ids)
    for key in evidence_keys:
        reference = registry.get(key)
        if key in seen or reference is None:
            continue
        seen.add(key)
        if reference["paper_id"] not in allowed_papers:
            continue
        if allowed_fields and reference["field"] not in allowed_fields:
            continue
        if not _valid_reference(reference):
            continue
        evidence.append({"evidence_key": key, **reference})
    return evidence


def _evidence_status(paper_ids: list[int], evidence: list[dict]) -> str:
    required = set(paper_ids)
    supported = {item["paper_id"] for item in evidence}
    if required and required.issubset(supported):
        return "fully_supported"
    if supported:
        return "partially_supported"
    return "unsupported"


def _normalize_title(title: str) -> set[str]:
    ignored = {"a", "an", "the", "of", "for", "in", "and", "limited",
               "lack", "insufficient"}
    return {
        token for token in re.findall(r"[a-z0-9]+", title.casefold())
        if token not in ignored
    }


def _duplicate_gap(title: str, accepted: list[dict]) -> bool:
    tokens = _normalize_title(title)
    for existing in accepted:
        other = _normalize_title(existing["title"])
        union = tokens | other
        if union and len(tokens & other) / len(union) >= 0.65:
            return True
    return False


def normalize_topic(text: str) -> set[str]:
    synonyms = {
        "patient": "clinical", "patients": "clinical",
        "histories": "history", "incorporation": "integration",
        "integrating": "integration", "cxr": "", "model": "",
        "models": "", "research": "", "area": ""
    }
    stopwords = {
        "a", "an", "the", "of", "for", "in", "and", "to", "with",
        "within", "additional", "underexplored", "limited"
    }
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        token = synonyms.get(token, token)
        if token and token not in stopwords:
            tokens.append(token)
    return set(tokens)


def _topic_overlap(first: str, second: str) -> float:
    left, right = normalize_topic(first), normalize_topic(second)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _normalized_confidence(
    requested: str,
    evidence: list[dict]
) -> str:
    explicit_papers = {
        item["paper_id"] for item in evidence
        if set(item["support_types"]) & {
            "explicit_limitation", "explicit_future_work"
        }
    }
    supporting_papers = {item["paper_id"] for item in evidence}
    if len(explicit_papers) >= 2:
        maximum = "high"
    elif len(explicit_papers) == 1 or len(supporting_papers) >= 2:
        maximum = "medium"
    else:
        maximum = "low"
    order = {"low": 0, "medium": 1, "high": 2}
    return requested if order[requested] <= order[maximum] else maximum


def build_evidence_diversity(evidence: list[dict]) -> dict:
    return {
        "paper_count": len({item["paper_id"] for item in evidence}),
        "evidence_count": len(evidence),
        "source_types": sorted({
            source_type for item in evidence
            for source_type in item["support_types"]
        }),
        "field_types": sorted({item["field"] for item in evidence})
    }


def calculate_gap_strength(evidence: list[dict]) -> str:
    diversity = build_evidence_diversity(evidence)
    sources = set(diversity["source_types"])
    explicit = bool(sources & {
        "explicit_limitation", "explicit_future_work"
    })
    contradictory = "potential_conflict" in sources
    if (
        diversity["paper_count"] >= 2
        and diversity["evidence_count"] >= 2
        and explicit
        and not contradictory
    ):
        return "strong"
    if explicit or (
        diversity["paper_count"] >= 2 and not contradictory
    ):
        return "moderate"
    return "tentative"


def build_gap_traceability(
    evidence: list[dict], requested_evidence_keys: list[str]
) -> dict:
    sources = {
        source_type for item in evidence
        for source_type in item["support_types"]
    }
    valid_keys = {item["evidence_key"] for item in evidence}
    requested_keys = set(requested_evidence_keys)
    paper_count = len({item["paper_id"] for item in evidence})
    comparison_types = {"cross_paper_difference", "comparison_caution"}
    inferred_types = comparison_types | {
        "potential_conflict", "dataset_pattern", "method_pattern", "other"
    }
    return {
        "valid_evidence": bool(evidence),
        "supporting_paper_count": paper_count,
        "evidence_item_count": len(evidence),
        "explicit_limitation_support": "explicit_limitation" in sources,
        "explicit_future_work_support": "explicit_future_work" in sources,
        "cross_paper_support": paper_count >= 2,
        "comparison_support": bool(sources & comparison_types),
        "conflict_support": "potential_conflict" in sources,
        "inferred_support": bool(sources & inferred_types),
        "unsupported_evidence_removed": len(requested_keys - valid_keys)
    }


def validate_candidate_gaps(
    gaps: list[CandidateGap],
    registry: dict[str, dict]
) -> list[dict]:
    validated = []
    for gap in gaps:
        if (
            not _has_text(gap.title)
            or not _has_text(gap.description)
            or not gap.supporting_paper_ids
            or _duplicate_gap(gap.title, validated)
        ):
            continue
        evidence = validate_gap_evidence(
            gap.evidence_keys, gap.supporting_paper_ids, registry
        )
        status = _evidence_status(gap.supporting_paper_ids, evidence)
        if status == "unsupported":
            continue
        support_types = {
            support_type
            for item in evidence
            for support_type in item["support_types"]
        }
        direct_signals = {
            "explicit_limitation", "explicit_future_work",
            "comparison_caution", "potential_conflict",
            "cross_paper_difference"
        }
        pattern_signals = {"dataset_pattern", "method_pattern"}
        if not (
            support_types & direct_signals
            or (
                support_types & pattern_signals
                and len({item["paper_id"] for item in evidence}) >= 2
            )
        ):
            continue
        requested_evidence_keys = list(dict.fromkeys(gap.evidence_keys))
        data = gap.model_dump()
        data.pop("evidence_keys", None)
        data["gap_id"] = f"GAP_{len(validated) + 1:03d}"
        data["confidence"] = _normalized_confidence(gap.confidence, evidence)
        supported_papers = sorted({item["paper_id"] for item in evidence})
        cross_paper = len(supported_papers) >= 2
        data["support_scope"] = (
            "cross_paper" if cross_paper else "single_paper"
        )
        data["cross_paper_support"] = cross_paper
        explicit_types = support_types & {
            "explicit_limitation", "explicit_future_work"
        }
        if cross_paper:
            data["support_summary"] = (
                f"Supported by {len(supported_papers)} papers"
                + (
                    " through explicit limitation or future-work evidence."
                    if explicit_types else " through coherent cross-paper evidence."
                )
            )
        else:
            data["support_summary"] = (
                "Supported by one paper through "
                + (
                    "explicit limitation or future-work evidence."
                    if explicit_types else "indirect evidence."
                )
            )
        data["evidence"] = evidence
        data["evidence_status"] = status
        data["evidence_valid"] = status == "fully_supported"
        data["evidence_diversity"] = build_evidence_diversity(evidence)
        data["gap_strength"] = calculate_gap_strength(evidence)
        data["traceability"] = build_gap_traceability(
            evidence, requested_evidence_keys
        )
        validated.append(data)
    return validated


def build_gap_evidence_matrix(
    gaps: list[dict], selected_paper_ids: list[int]
) -> list[dict]:
    matrix = []
    for gap in gaps:
        paper_support = []
        for paper_id in selected_paper_ids:
            evidence = [
                item for item in gap["evidence"]
                if item["paper_id"] == paper_id
            ]
            paper_support.append({
                "paper_id": paper_id,
                "support": bool(evidence),
                "support_types": sorted({
                    source_type for item in evidence
                    for source_type in item["support_types"]
                }),
                "evidence_keys": [
                    item["evidence_key"] for item in evidence
                ]
            })
        matrix.append({
            "gap_id": gap["gap_id"],
            "gap_title": gap["title"],
            "paper_support": paper_support
        })
    return matrix


def build_recurring_limitations(
    limitations: list[RecurringLimitation],
    registry: dict[str, dict]
) -> list[dict]:
    output = []
    for limitation in limitations:
        paper_ids = list(dict.fromkeys(limitation.paper_ids))
        if len(paper_ids) < 2:
            continue
        evidence = validate_gap_evidence(
            limitation.evidence_keys, paper_ids, registry, {"limitations"}
        )
        status = _evidence_status(paper_ids, evidence)
        if status != "fully_supported":
            continue
        data = limitation.model_dump(exclude={"evidence_keys"})
        data.update({"evidence": evidence, "evidence_valid": True,
                     "evidence_status": status})
        output.append(data)
    return output


def _validate_auxiliary(
    items: list[BaseModel],
    registry: dict[str, dict],
    paper_id_field: str
) -> list[dict]:
    output = []
    for item in items:
        data = item.model_dump()
        paper_ids = list(dict.fromkeys(data[paper_id_field]))
        evidence = validate_gap_evidence(
            data.pop("evidence_keys", []), paper_ids, registry
        )
        status = _evidence_status(paper_ids, evidence)
        if status == "unsupported":
            continue
        data[paper_id_field] = paper_ids
        data.update({"evidence": evidence,
                     "evidence_valid": status == "fully_supported",
                     "evidence_status": status})
        output.append(data)
    return output


def deduplicate_underexplored_areas(
    areas: list[dict],
    candidate_gaps: Optional[list[dict]] = None
) -> list[dict]:
    """Merge strongly overlapping areas with matching evidence."""

    merged = []
    for area in areas:
        area_keys = {item["evidence_key"] for item in area["evidence"]}
        duplicate = None
        for existing in merged:
            existing_keys = {
                item["evidence_key"] for item in existing["evidence"]
            }
            if (
                area_keys == existing_keys
                and _topic_overlap(area["area"], existing["area"]) >= 0.6
            ):
                duplicate = existing
                break
        if duplicate is None:
            merged.append(area)
        else:
            duplicate["paper_ids"] = list(dict.fromkeys(
                duplicate["paper_ids"] + area["paper_ids"]
            ))
            if area["basis"] not in duplicate["basis"]:
                duplicate["basis"] += " " + area["basis"]

    distinct = []
    for area in merged:
        area_keys = {item["evidence_key"] for item in area["evidence"]}
        duplicates_gap = any(
            area_keys == {item["evidence_key"] for item in gap["evidence"]}
            and _topic_overlap(area["area"], gap["title"]) >= 0.75
            for gap in (candidate_gaps or [])
        )
        if not duplicates_gap:
            distinct.append(area)
    return distinct


def build_future_direction_signals(
    signals: list[FutureDirectionSignal],
    registry: dict[str, dict]
) -> list[dict]:
    return _validate_auxiliary(signals, registry, "paper_ids")


def _generation_basis(evidence: list[dict]) -> str:
    sources = {
        source_type for item in evidence
        for source_type in item["support_types"]
    }
    if "explicit_future_work" in sources:
        return "explicit_future_work"
    if "explicit_limitation" in sources:
        return "explicit_limitation"
    if "potential_conflict" in sources:
        return "conflict_resolution"
    return "cross_paper_inference"


def _extension_evidence(
    gap: dict, evidence_keys: list[str], registry: dict[str, dict]
) -> list[dict]:
    allowed_keys = {item["evidence_key"] for item in gap["evidence"]}
    permitted_keys = [
        key for key in evidence_keys if key in allowed_keys
    ]
    return validate_gap_evidence(
        permitted_keys, gap["supporting_paper_ids"], registry
    )


def validate_research_questions(
    questions: list[ResearchQuestionLLM],
    gaps: list[dict],
    registry: dict[str, dict]
) -> list[dict]:
    gap_map = {gap["gap_id"]: gap for gap in gaps}
    accepted, counts = [], Counter()
    for question in questions:
        gap = gap_map.get(question.gap_id)
        if gap is None or not _has_text(question.question):
            continue
        evidence = _extension_evidence(
            gap, question.evidence_keys, registry
        )
        if not evidence:
            continue
        maximum = 3 if gap["gap_strength"] == "strong" else 1
        if counts[gap["gap_id"]] >= maximum:
            continue
        if any(
            item["gap_id"] == gap["gap_id"]
            and _topic_overlap(item["question"], question.question) >= 0.8
            for item in accepted
        ):
            continue
        paper_ids = sorted({item["paper_id"] for item in evidence})
        accepted.append({
            "question_id": f"RQ_{len(accepted) + 1:03d}",
            "gap_id": gap["gap_id"],
            "question": question.question.strip(),
            "question_type": question.question_type,
            "generation_basis": _generation_basis(evidence),
            "supporting_paper_ids": paper_ids,
            "evidence_keys": [item["evidence_key"] for item in evidence],
            "evidence": evidence,
            "evidence_valid": True,
            "evidence_status": "fully_supported"
        })
        counts[gap["gap_id"]] += 1
    return accepted


def validate_research_opportunities(
    opportunities: list[ResearchOpportunityLLM],
    gaps: list[dict],
    registry: dict[str, dict]
) -> list[dict]:
    gap_map = {gap["gap_id"]: gap for gap in gaps}
    accepted, counts = [], Counter()
    for opportunity in opportunities:
        gap = gap_map.get(opportunity.gap_id)
        if (
            gap is None
            or not _has_text(opportunity.title)
            or not _has_text(opportunity.description)
        ):
            continue
        evidence = _extension_evidence(
            gap, opportunity.evidence_keys, registry
        )
        if not evidence:
            continue
        maximum = 2 if gap["gap_strength"] == "strong" else 1
        if counts[gap["gap_id"]] >= maximum:
            continue
        if any(
            item["gap_id"] == gap["gap_id"]
            and _topic_overlap(item["title"], opportunity.title) >= 0.8
            for item in accepted
        ):
            continue
        paper_ids = sorted({item["paper_id"] for item in evidence})
        accepted.append({
            "opportunity_id": f"OPP_{len(accepted) + 1:03d}",
            "gap_id": gap["gap_id"],
            "title": opportunity.title.strip(),
            "description": opportunity.description.strip(),
            "opportunity_type": opportunity.opportunity_type,
            "generation_basis": _generation_basis(evidence),
            "confidence": gap["confidence"],
            "supporting_paper_ids": paper_ids,
            "evidence_keys": [item["evidence_key"] for item in evidence],
            "evidence": evidence,
            "evidence_valid": True,
            "evidence_status": "fully_supported"
        })
        counts[gap["gap_id"]] += 1
    return accepted


def deduplicate_research_opportunities(
    opportunities: list[dict], underexplored_areas: list[dict]
) -> list[dict]:
    """Keep proposed actions distinct from literature observations."""

    area_texts = {
        re.sub(r"\s+", " ", area["area"].casefold()).strip()
        for area in underexplored_areas
        if _has_text(area.get("area"))
    }
    return [
        opportunity for opportunity in opportunities
        if re.sub(
            r"\s+", " ", opportunity["title"].casefold()
        ).strip() not in area_texts
        and re.sub(
            r"\s+", " ", opportunity["description"].casefold()
        ).strip() not in area_texts
    ]


def generate_gap_extensions(
    gaps: list[dict], registry: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Generate all question and opportunity candidates in one LLM call."""

    if not gaps:
        return [], []
    context = [{
        "gap_id": gap["gap_id"],
        "title": gap["title"],
        "description": gap["description"],
        "gap_type": gap["gap_type"],
        "confidence": gap["confidence"],
        "gap_strength": gap["gap_strength"],
        "support_scope": gap["support_scope"],
        "evidence_keys": [
            item["evidence_key"] for item in gap["evidence"]
        ],
        "evidence": gap["evidence"]
    } for gap in gaps]
    prompt = f"""
Using ONLY these validated candidate gaps and their evidence, generate neutral,
testable research questions and concrete research opportunities. Generate one
question and one opportunity for a single-paper or tentative/moderate gap. A
strong cross-paper gap may have up to three questions and two opportunities.
Return all gaps in this ONE response; do not invent or alter gap IDs.

Every item must use only evidence_keys belonging to its gap and must include at
least one key. Questions are generated research directions, not paper findings.
Opportunities are proposed investigations, not conclusions from the papers.
Do not claim that no research exists, do not use missing_information or
evidence_issues, and do not invent datasets, methods, metrics, limitations, or
future work. Prefer explicit future-work evidence when it is available. Use only
the allowed normalized question, opportunity, and generation-basis categories.

Validated gaps:
{json.dumps(context, ensure_ascii=False, separators=(",", ":"))}
"""
    draft = generate_structured_answer(
        prompt=prompt, response_schema=GapExtensionLLMOutput
    )
    if not isinstance(draft, GapExtensionLLMOutput):
        draft = GapExtensionLLMOutput.model_validate(draft)
    return (
        validate_research_questions(
            draft.research_questions, gaps, registry
        ),
        validate_research_opportunities(
            draft.research_opportunities, gaps, registry
        )
    )


def calculate_gap_evidence_coverage(gaps: list[dict]) -> dict:
    counts = Counter(gap["evidence_status"] for gap in gaps)
    total = len(gaps)
    return {
        "total_gaps": total,
        "fully_supported": counts["fully_supported"],
        "partially_supported": counts["partially_supported"],
        "unsupported": counts["unsupported"],
        "coverage_ratio": round(
            counts["fully_supported"] / total if total else 0.0,
            3
        )
    }


def build_citation(evidence: dict) -> str:
    return (
        f"[Paper {evidence['paper_id']}, "
        f"p.{evidence['page_number']}, {evidence['chunk_id']}]"
    )


def extract_numeric_tokens(text: str) -> set[str]:
    without_paper_labels = re.sub(
        r"\bpaper\s+\d+\b", "", text, flags=re.IGNORECASE
    )
    return set(re.findall(
        r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?",
        without_paper_labels
    ))


def validate_report_claim(
    claim: ReportClaim,
    registry: dict[str, dict]
) -> Optional[str]:
    evidence_papers = [
        registry[key]["paper_id"]
        for key in claim.evidence_keys
        if key in registry and _valid_reference(registry[key])
    ]
    paper_ids = list(dict.fromkeys(claim.paper_ids or evidence_papers))
    evidence = validate_gap_evidence(
        claim.evidence_keys, paper_ids, registry
    )
    if _evidence_status(paper_ids, evidence) != "fully_supported":
        return None

    claim_numbers = extract_numeric_tokens(claim.text)
    evidence_numbers = set().union(*(
        extract_numeric_tokens(
            f"{item['value']} {item['evidence_text']}"
        )
        for item in evidence
    )) if evidence else set()
    if not claim_numbers.issubset(evidence_numbers):
        return None

    citations = list(dict.fromkeys(
        build_citation(item) for item in evidence
    ))
    return f"{claim.text.strip()} {' '.join(citations)}".strip()


def build_report_context(
    cross_paper_output: dict,
    gaps: dict,
    registry: dict[str, dict]
) -> str:
    return json.dumps({
        "cross_paper_analysis": {
            key: value for key, value in cross_paper_output.items()
            if key not in {"missing_information", "evidence_issues"}
        },
        "validated_agent4_findings": gaps,
        "evidence_registry": registry
    }, ensure_ascii=False, separators=(",", ":"))


def generate_research_report(
    cross_paper_output: dict,
    validated_findings: dict,
    registry: dict[str, dict]
) -> dict:
    context = build_report_context(
        cross_paper_output, validated_findings, registry
    )
    prompt = f"""
Write a structured academic literature synthesis using ONLY the supplied data.
Return each section as separate factual claims. Every claim must list its exact
evidence_keys and all specific paper_ids it discusses. Never invent keys,
citations, paper facts, gaps, limitations, future work, numbers, or conversions.
For numeric claims, use the exact textual value from the cited evidence.

Scope every conclusion to "within the analyzed papers", "the selected studies",
"the reviewed paper set", or "the available evidence". Never say the field has
proven, established, or completely lacks something. Omit unsupported claims.
Do not use missing_information or evidence_issues as scientific evidence.
Introduce research questions and research opportunities as generated synthesis
based on validated candidate gaps. Never imply that the papers themselves state
or conclude a generated question or proposed opportunity.

Context:
{context}
"""
    draft = generate_structured_answer(
        prompt=prompt,
        response_schema=ResearchReportLLMOutput
    )
    if not isinstance(draft, ResearchReportLLMOutput):
        draft = ResearchReportLLMOutput.model_validate(draft)
    report = {"title": draft.title}
    for field_name, section in draft.model_dump().items():
        if field_name == "title":
            continue
        rendered_claims = []
        for claim_data in section["claims"]:
            rendered = validate_report_claim(
                ReportClaim.model_validate(claim_data), registry
            )
            if rendered:
                rendered_claims.append(rendered)
        report[field_name] = "\n\n".join(rendered_claims)
    return report


def analyze_research_gaps(
    papers: list[Any],
    cross_paper_output: dict
) -> dict:
    if len(papers) < 2:
        raise ValueError("At least two analyzed papers are required.")
    registry = build_agent4_evidence_registry(papers, cross_paper_output)
    context = build_gap_context(papers, cross_paper_output, registry)
    prompt = f"""
You are the Research Gap and Report Agent. Analyze ONLY the supplied Agent 2 and
Agent 3 data. A missing topic in selected papers is not automatically a gap.
Never claim absence from the entire field. Use cautious phrases such as candidate
research gap, within the analyzed paper set, and the evidence suggests.

Do not use outside knowledge. Do not invent evidence keys, limitations, future
work, or conflicts. Do not turn missing Agent 2 fields, citation failures, or
evidence issues into gaps. A different architecture or metric alone is not a gap.
A performance difference is not a gap unless it represents an evidenced unresolved
issue. Do not create a gap solely because one paper uses a method another does not.

High confidence requires related explicit limitations/future work from multiple
papers. Medium confidence requires two coherent indirect sources or one explicit
important limitation/future direction. Otherwise use low confidence. Recurring
limitations require valid limitation evidence from at least two papers. Create
unresolved conflicts only from Agent 3 potential_conflicts. Label future directions
as explicit_future_work, explicit_limitation, or cross_paper_inference.
Every item must cite only supplied evidence_keys.

Use only these gap_type values: dataset_gap, methodological_gap, evaluation_gap,
generalization_gap, clinical_validation_gap, modality_gap, clinical_context_gap,
annotation_gap, benchmark_gap, task_coverage_gap, robustness_gap,
interpretability_gap, deployment_gap, efficiency_gap, scaling_gap,
architecture_gap, or other. A gap supported by one paper is a single-paper signal,
not a field-wide conclusion. Use conservative language scoped to the selected set.

Context:
{context}
"""
    output = generate_structured_answer(
        prompt=prompt,
        response_schema=ResearchGapLLMOutput
    )
    if not isinstance(output, ResearchGapLLMOutput):
        output = ResearchGapLLMOutput.model_validate(output)

    candidate_gaps = validate_candidate_gaps(output.candidate_gaps, registry)
    recurring = build_recurring_limitations(
        output.recurring_limitations, registry
    )
    underexplored = deduplicate_underexplored_areas(
        _validate_auxiliary(
            output.underexplored_areas, registry, "paper_ids"
        ),
        candidate_gaps
    )
    conflicts = (
        _validate_auxiliary(
            output.unresolved_conflicts, registry, "paper_ids"
        )
        if cross_paper_output.get("potential_conflicts")
        else []
    )
    for conflict in conflicts:
        conflict["requires_further_study"] = True
    future_signals = build_future_direction_signals(
        output.future_direction_signals, registry
    )
    selected_paper_ids = [int(_value(paper, "paper_id")) for paper in papers]
    gap_evidence_matrix = build_gap_evidence_matrix(
        candidate_gaps, selected_paper_ids
    )
    research_questions, research_opportunities = generate_gap_extensions(
        candidate_gaps, registry
    )
    research_opportunities = deduplicate_research_opportunities(
        research_opportunities, underexplored
    )
    validated_findings = {
        "candidate_gaps": candidate_gaps,
        "research_questions": research_questions,
        "research_opportunities": research_opportunities,
        "gap_evidence_matrix": gap_evidence_matrix,
        "recurring_limitations": recurring,
        "underexplored_areas": underexplored,
        "unresolved_conflicts": conflicts,
        "future_direction_signals": future_signals
    }
    support_counts = Counter(
        support_type
        for gap in candidate_gaps
        for evidence in gap["evidence"]
        for support_type in evidence["support_types"]
    )
    gap_summary = {
        "total_candidate_gaps": len(candidate_gaps),
        "strong_gaps": sum(
            gap["gap_strength"] == "strong" for gap in candidate_gaps
        ),
        "moderate_gaps": sum(
            gap["gap_strength"] == "moderate" for gap in candidate_gaps
        ),
        "tentative_gaps": sum(
            gap["gap_strength"] == "tentative" for gap in candidate_gaps
        ),
        "cross_paper_gaps": sum(
            gap["support_scope"] == "cross_paper"
            for gap in candidate_gaps
        ),
        "single_paper_gaps": sum(
            gap["support_scope"] == "single_paper"
            for gap in candidate_gaps
        ),
        "generated_research_questions": len(research_questions),
        "generated_research_opportunities": len(research_opportunities),
        "support_type_counts": dict(support_counts),
        "supporting_paper_ids": sorted({
            paper_id for gap in candidate_gaps
            for paper_id in gap["supporting_paper_ids"]
        })
    }
    report = generate_research_report(
        cross_paper_output, validated_findings, registry
    )
    return {
        "paper_ids": selected_paper_ids,
        **validated_findings,
        "gap_evidence_summary": gap_summary,
        "report": report,
        "evidence_coverage": calculate_gap_evidence_coverage(candidate_gaps)
    }

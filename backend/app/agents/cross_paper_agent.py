import json
import re
from itertools import combinations
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.services.llm_service import generate_structured_answer


ComparisonField = Literal[
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations",
    "future_work"
]
ConflictType = Literal[
    "dataset_difference", "model_difference",
    "population_difference", "training_difference",
    "evaluation_difference", "metric_difference",
    "genuine_finding_difference", "unclear"
]
COMPARISON_FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations",
    "future_work"
)


class ComparisonFinding(BaseModel):
    statement: str
    paper_ids: list[int] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    evidence_fields: list[ComparisonField] = Field(default_factory=list)


class ComparisonCaution(BaseModel):
    topic: str
    paper_ids: list[int] = Field(default_factory=list)
    statement: str
    reason: str
    evidence_keys: list[str] = Field(default_factory=list)
    evidence_fields: list[ComparisonField] = Field(default_factory=list)


class ComparabilityAssessment(BaseModel):
    paper_ids: list[int]
    dataset_comparable: bool
    task_comparable: bool
    metric_comparable: bool
    population_comparable: Optional[bool] = None
    evaluation_setup_comparable: bool
    direct_result_comparison_allowed: bool
    reason: str


class PotentialConflict(BaseModel):
    topic: str
    paper_ids: list[int] = Field(default_factory=list)
    description: str
    explanation: Optional[str] = None
    conflict_type: ConflictType = "unclear"
    evidence_keys: list[str] = Field(default_factory=list)
    evidence_fields: list[ComparisonField] = Field(
        default_factory=lambda: ["main_results"]
    )


class CrossPaperLLMOutput(BaseModel):
    similarities: list[ComparisonFinding] = Field(default_factory=list)
    differences: list[ComparisonFinding] = Field(default_factory=list)
    comparison_cautions: list[ComparisonCaution] = Field(default_factory=list)
    comparability_assessment: list[ComparabilityAssessment] = Field(
        default_factory=list
    )
    common_limitations: list[ComparisonFinding] = Field(default_factory=list)
    potential_conflicts: list[PotentialConflict] = Field(default_factory=list)
    overall_observations: list[str] = Field(default_factory=list)


def get_field_value(field_data: Any):
    if isinstance(field_data, dict):
        return field_data.get("value")
    return getattr(field_data, "value", None)


def _paper_value(paper: Any, name: str):
    if isinstance(paper, dict):
        return paper.get(name)
    return getattr(paper, name, None)


def _has_text(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _is_valid_evidence(reference: dict) -> bool:
    return (
        reference["citation_valid"]
        and _has_text(reference["value"])
        and _has_text(reference["evidence_text"])
        and reference["page_number"] is not None
        and _has_text(reference["chunk_id"])
    )


def create_evidence_registry(papers: list[Any]) -> dict[str, dict]:
    registry = {}
    for paper in papers:
        paper_id = int(_paper_value(paper, "paper_id"))
        analysis = _paper_value(paper, "analysis") or {}
        for field_name in COMPARISON_FIELDS:
            field_data = _paper_value(analysis, field_name) or {}
            registry[f"P{paper_id}_{field_name}"] = {
                "paper_id": paper_id,
                "field": field_name,
                "value": get_field_value(field_data),
                "evidence_text": _paper_value(field_data, "evidence_text"),
                "page_number": _paper_value(field_data, "page_number"),
                "chunk_id": _paper_value(field_data, "chunk_id"),
                "citation_valid": bool(
                    _paper_value(field_data, "citation_valid")
                )
            }
    return registry


def build_comparison_table(papers: list[Any]) -> list[dict]:
    table = []
    for paper in papers:
        analysis = _paper_value(paper, "analysis") or {}
        row = {
            "paper_id": int(_paper_value(paper, "paper_id")),
            "title": _paper_value(paper, "title"),
            "year": _paper_value(paper, "year")
        }
        for field_name in COMPARISON_FIELDS:
            row[field_name] = get_field_value(
                _paper_value(analysis, field_name)
            )
        table.append(row)
    return table


def build_cross_paper_context(
    comparison_table: list[dict], evidence_registry: dict[str, dict]
) -> str:
    return json.dumps(
        {"comparison_table": comparison_table,
         "evidence_registry": evidence_registry},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def validate_evidence_keys(
    evidence_keys: list[str], evidence_registry: dict[str, dict],
    paper_ids: Optional[list[int]] = None,
    allowed_fields: Optional[list[str]] = None
) -> list[dict]:
    allowed_papers = set(paper_ids) if paper_ids is not None else None
    allowed = set(allowed_fields) if allowed_fields else None
    evidence, seen = [], set()
    for key in evidence_keys:
        reference = evidence_registry.get(key)
        if key in seen or reference is None:
            continue
        seen.add(key)
        if allowed_papers is not None and reference["paper_id"] not in allowed_papers:
            continue
        if allowed is not None and reference["field"] not in allowed:
            continue
        evidence.append({"evidence_key": key, **reference})
    return evidence


def _infer_fields(keys: list[str], registry: dict[str, dict]) -> list[str]:
    for key in keys:
        if key in registry:
            return [registry[key]["field"]]
    return []


def _apply_evidence_status(
    data: dict,
    evidence: list[dict],
    required_fields: Optional[list[str]] = None
) -> dict:
    required = set(data["paper_ids"])
    required_field_names = set(required_fields or [])
    supported = {
        item["paper_id"] for item in evidence if _is_valid_evidence(item)
    }
    supported_pairs = {
        (item["paper_id"], item["field"])
        for item in evidence
        if _is_valid_evidence(item)
    }
    required_pairs = {
        (paper_id, field_name)
        for paper_id in required
        for field_name in required_field_names
    }
    has_full_field_coverage = (
        not required_pairs
        or required_pairs.issubset(supported_pairs)
    )
    if (
        required
        and required.issubset(supported)
        and has_full_field_coverage
    ):
        status = "fully_supported"
    elif supported:
        status = "partially_supported"
    else:
        status = "unsupported"
    data.update({
        "evidence": evidence,
        "evidence_valid": status == "fully_supported",
        "evidence_status": status
    })
    return data


def validate_findings(
    findings: list[ComparisonFinding], registry: dict[str, dict],
    required_fields: Optional[list[str]] = None
) -> list[dict]:
    validated = []
    for finding in findings:
        data = finding.model_dump()
        fields = required_fields or finding.evidence_fields or _infer_fields(
            finding.evidence_keys, registry
        )
        evidence = validate_evidence_keys(
            finding.evidence_keys, registry, finding.paper_ids, fields
        )
        data.pop("evidence_keys", None)
        data.pop("evidence_fields", None)
        validated.append(_apply_evidence_status(data, evidence, fields))
    return validated


def validate_cautions(
    cautions: list[ComparisonCaution], registry: dict[str, dict]
) -> list[dict]:
    validated = []
    for caution in cautions:
        data = caution.model_dump()
        paper_ids = list(dict.fromkeys(caution.paper_ids))
        if not paper_ids:
            paper_ids = list(dict.fromkeys(
                registry[key]["paper_id"]
                for key in caution.evidence_keys
                if key in registry
                and _is_valid_evidence(registry[key])
            ))
        if len(paper_ids) < 2:
            continue
        data["paper_ids"] = paper_ids
        fields = caution.evidence_fields or list(dict.fromkeys(
            registry[key]["field"]
            for key in caution.evidence_keys
            if key in registry
        ))
        evidence = validate_evidence_keys(
            caution.evidence_keys, registry, paper_ids, fields
        )
        data.pop("evidence_keys", None)
        data.pop("evidence_fields", None)
        validated.append(_apply_evidence_status(data, evidence, fields))
    return validated


def validate_comparability_assessments(
    assessments: list[ComparabilityAssessment], paper_ids: list[int]
) -> list[dict]:
    selected, supplied = set(paper_ids), {}
    for assessment in assessments:
        pair = tuple(sorted(set(assessment.paper_ids)))
        if len(pair) == 2 and set(pair).issubset(selected):
            data = assessment.model_dump()
            data["paper_ids"] = list(pair)
            supplied[pair] = data
    result = []
    for pair in combinations(paper_ids, 2):
        canonical = tuple(sorted(pair))
        result.append(supplied.get(canonical, {
            "paper_ids": list(pair),
            "dataset_comparable": False,
            "task_comparable": False,
            "metric_comparable": False,
            "population_comparable": None,
            "evaluation_setup_comparable": False,
            "direct_result_comparison_allowed": False,
            "reason": "The structured data was insufficient to establish direct comparability."
        }))
    return result


def validate_conflicts(
    conflicts: list[PotentialConflict], registry: dict[str, dict],
    assessments: Optional[list[dict]] = None
) -> list[dict]:
    comparable_pairs = {
        tuple(sorted(item["paper_ids"])) for item in (assessments or [])
        if item["direct_result_comparison_allowed"]
    }
    validated = []
    for conflict in conflicts:
        pairs = {
            tuple(sorted(pair))
            for pair in combinations(sorted(set(conflict.paper_ids)), 2)
        }
        if assessments is not None and pairs and not pairs.issubset(comparable_pairs):
            continue
        data = conflict.model_dump()
        fields = conflict.evidence_fields or ["main_results"]
        evidence = validate_evidence_keys(
            conflict.evidence_keys, registry, conflict.paper_ids, fields
        )
        data.pop("evidence_keys", None)
        data.pop("evidence_fields", None)
        validated.append(_apply_evidence_status(data, evidence, fields))
    return validated


def detect_missing_information(
    papers: list[Any], registry: dict[str, dict]
) -> list[dict]:
    output = []
    for paper in papers:
        paper_id = int(_paper_value(paper, "paper_id"))
        missing = []
        for field in COMPARISON_FIELDS:
            ref = registry[f"P{paper_id}_{field}"]
            has_evidence = _is_valid_evidence(ref)
            if not _has_text(ref["value"]) or not has_evidence:
                missing.append(field)
        if missing:
            output.append({"paper_id": paper_id, "missing_fields": missing})
    return output


DATASET_ALIASES = (
    (r"\bMIMIC[- ]?CXR[- ]?JPG\b", "MIMIC-CXR-JPG"),
    (r"\bMIMIC[- ]?CXR\b(?![- ]?JPG)", "MIMIC-CXR"),
    (r"\b(?:NIH[ -]+)?Chest[- ]?X[- ]?ray[- ]?14\b", "ChestX-ray14"),
    (r"\bCheXpert\b", "CheXpert"),
    (r"\bPadChest\b", "PadChest"),
    (r"\bVinDr[- ]?CXR\b", "VinDr-CXR"),
    (r"\bRSNA Pneumonia Detection Challenge\b",
     "RSNA Pneumonia Detection Challenge")
)


def extract_dataset_names(value: str) -> list[str]:
    """Extract known dataset names conservatively from Agent 2 text."""

    matches = []
    for pattern, canonical_name in DATASET_ALIASES:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            matches.append((match.start(), canonical_name))
    if matches:
        return list(dict.fromkeys(
            name for _, name in sorted(matches)
        ))
    return [" ".join(value.split())]


def build_dataset_analysis(table: list[dict], registry: dict[str, dict]) -> dict:
    grouped, raw = {}, []
    for row in table:
        value = row.get("dataset")
        if not _has_text(value):
            continue
        raw.append({"paper_id": row["paper_id"], "value": value})
        for name in extract_dataset_names(str(value)):
            entry = grouped.setdefault(
                name.casefold(),
                {"name": name, "paper_ids": [], "evidence": []}
            )
            if row["paper_id"] not in entry["paper_ids"]:
                entry["paper_ids"].append(row["paper_id"])
                key = f"P{row['paper_id']}_dataset"
                entry["evidence"].append({
                    "evidence_key": key,
                    **registry[key]
                })
    return {"datasets": list(grouped.values()), "raw_values": raw}


METRIC_PATTERNS = (
    ("AUROC", r"\b(?:AUROC|AUC|ROC[- ]?AUC|area under (?:the )?ROC curve)\b"),
    ("AUPRC", r"\b(?:AUPRC|PR[- ]?AUC|area under (?:the )?precision[- ]recall curve)\b"),
    ("F1", r"\bF[- ]?1(?: score)?\b"),
    ("MCC", r"\b(?:MCC|Matthews correlation coefficient)\b"),
    ("Accuracy", r"\baccuracy\b"), ("Precision", r"\bprecision\b"),
    ("Recall", r"\brecall\b"), ("Sensitivity", r"\bsensitivity\b"),
    ("Specificity", r"\bspecificity\b"), ("BLEU", r"\bBLEU\b"),
    ("ROUGE", r"\bROUGE(?:-[A-Z0-9]+)?\b"),
    ("CIDEr", r"\bCIDEr\b"), ("METEOR", r"\bMETEOR\b"),
    ("BERTScore", r"\bBERTScore\b")
)


def extract_metric_names(value: str) -> list[str]:
    """Extract reported metrics while excluding explanatory definitions."""

    searchable = re.sub(
        r"(?:which\s+is\s+)?(?:the\s+)?harmonic\s+"
        r"(?:average|mean)\s+of\s+precision\s+and\s+recall",
        "",
        value,
        flags=re.IGNORECASE
    )
    matches = []
    for name, pattern in METRIC_PATTERNS:
        match = re.search(pattern, searchable, re.IGNORECASE)
        if match:
            matches.append((match.start(), name))
    return list(dict.fromkeys(
        name for _, name in sorted(matches)
    ))


def build_metric_analysis(table: list[dict], registry: dict[str, dict]) -> dict:
    grouped, raw = {}, []
    for row in table:
        value = row.get("evaluation_metrics")
        if not _has_text(value):
            continue
        raw.append({"paper_id": row["paper_id"], "value": value})
        names = extract_metric_names(str(value))
        for name in names:
            entry = grouped.setdefault(
                name.casefold(), {"name": name, "paper_ids": [], "evidence": []}
            )
            if row["paper_id"] not in entry["paper_ids"]:
                entry["paper_ids"].append(row["paper_id"])
                key = f"P{row['paper_id']}_evaluation_metrics"
                entry["evidence"].append({"evidence_key": key, **registry[key]})
    return {"metrics": list(grouped.values()), "raw_values": raw}


def classify_method_type(methodology: Any, model: Any) -> str:
    """Map Agent 2 method/model text to a short factual category."""

    text = f"{methodology or ''} {model or ''}".casefold()
    explicit_vision_language = any(term in text for term in (
        "image-text", "image text", "vision-language", "vision language",
        "language-image", "language image", "image-report", "image report",
        "paired image and text", "paired images and text"
    ))
    image_modality = any(term in text for term in (
        "image encoder", "visual encoder", "vision encoder",
        "vision transformer", "medical image", "radiograph", "x-ray"
    ))
    text_modality = any(term in text for term in (
        "text encoder", "language encoder", "textual encoder",
        "radiology report", "radiology reports", "clinical report",
        "text modality", "textual modality"
    ))
    multimodal = (
        explicit_vision_language
        or (image_modality and text_modality)
    )

    foundation_model = any(term in text for term in (
        "foundation model", "large vision-language model",
        "large vision language model"
    ))
    broad_multitask_pretraining = (
        multimodal
        and any(term in text for term in (
            "large-scale", "large scale", "pretrained", "pre-trained"
        ))
        and any(term in text for term in (
            "multiple tasks", "multi-task", "multitask",
            "range of tasks", "several tasks"
        ))
    )
    if (foundation_model and multimodal) or broad_multitask_pretraining:
        return "Vision-language foundation model"
    if "contrastive" in text and multimodal:
        return "Contrastive vision-language pretraining"
    if "self-supervised" in text and multimodal:
        return "Vision-language self-supervised learning"
    if "zero-shot" in text and multimodal:
        return "Zero-shot vision-language model"
    if multimodal:
        return "Vision-language model"

    cnn_model = any(term in text for term in (
        "densenet", "dense connections", "resnet", "efficientnet",
        "cnn", "convolutional", "convolution"
    ))
    supervised = any(term in text for term in (
        "supervised learning", "supervised training", "trained supervised",
        "labeled training", "labelled training"
    ))
    if cnn_model and supervised:
        return "Supervised CNN"
    if any(term in text for term in (
        "transformer", "vision transformer", "vit"
    )):
        return "Transformer-based classifier"
    if cnn_model:
        return "CNN classifier"
    return "Other / Unclassified"


def build_method_trends(table: list[dict], registry: dict[str, dict]) -> list[dict]:
    trends = []
    for row in table:
        methodology, model = row.get("methodology"), row.get("model")
        if not _has_text(methodology) and not _has_text(model):
            continue
        keys = [
            f"P{row['paper_id']}_{field}" for field in ("methodology", "model")
            if _has_text(registry[f"P{row['paper_id']}_{field}"]["value"])
        ]
        parts = [str(value) for value in (methodology, model)
                 if _has_text(value)]
        trends.append({
            "year": row.get("year"), "paper_id": row["paper_id"],
            "method_type": classify_method_type(methodology, model),
            "description": " ".join(dict.fromkeys(parts)),
            "evidence": validate_evidence_keys(
                keys, registry, [row["paper_id"]], ["methodology", "model"]
            )
        })
    return sorted(
        trends,
        key=lambda item: (item["year"] is None, item["year"] or 0, item["paper_id"])
    )


def calculate_evidence_coverage(*groups: list[dict]) -> dict:
    findings = [item for group in groups for item in group]
    counts = {
        status: sum(item.get("evidence_status") == status for item in findings)
        for status in ("fully_supported", "partially_supported", "unsupported")
    }
    total = len(findings)
    return {
        "total_findings": total, **counts,
        "coverage_ratio": round(
            counts["fully_supported"] / total if total else 0.0,
            3
        )
    }


def compare_papers(papers: list[Any]) -> dict:
    if len(papers) < 2:
        raise ValueError("At least two analyzed papers are required.")

    table = build_comparison_table(papers)
    registry = create_evidence_registry(papers)
    context = build_cross_paper_context(table, registry)
    prompt = f"""
You are the Cross-Paper Analysis Agent in an evidence-grounded research system.
Use only the supplied structured Agent 2 data. Do not use outside knowledge,
invent evidence keys, or invent missing paper information.

Rules:
1. Every claim and caution must cite evidence from every named paper where possible.
2. Set evidence_fields to the exact structured fields used. Use field-matching
   evidence only. Never use main_results for objectives or datasets for limitations.
3. Missing information is unknown, not negative evidence. Omit unsupported claims.
4. Put non-comparable results under comparison_cautions, never potential_conflicts.
5. Assess every paper pair for dataset, task, metric, population, evaluation setup,
   and direct-result comparability using booleans and a reason, never a score.
6. A potential conflict requires similar questions, meaningfully different findings,
   and main-results evidence from every paper. Use genuine_finding_difference only
   when the papers are sufficiently comparable; otherwise explain contextual causes.
7. Explain dataset, task, metric, population, training, or evaluation differences.
8. Do not rank papers solely from reported numbers. Do not generate research gaps.
9. Focus on meaningful research comparisons, not superficial wording differences.
10. Do not compare a field when any referenced paper has no value for that field.
11. Never interpret missing information as a scientific difference or claim that a
    paper has no objective, limitation, or future work merely because it is missing.
12. Do not return a comparison caution without all involved paper IDs and matching
    evidence from every involved paper.
13. Omit unsupported similarities and any claim that lacks valid keys from every
    paper it names.
14. Preserve the distinction between a difference, comparison caution, and
    potential conflict.

Structured paper data and evidence registry:
{context}
"""
    output = generate_structured_answer(
        prompt=prompt, response_schema=CrossPaperLLMOutput
    )
    if not isinstance(output, CrossPaperLLMOutput):
        output = CrossPaperLLMOutput.model_validate(output)

    paper_ids = [row["paper_id"] for row in table]
    similarities = [
        item for item in validate_findings(output.similarities, registry)
        if item["evidence_status"] == "fully_supported"
    ]
    differences = [
        item for item in validate_findings(output.differences, registry)
        if item["evidence_status"] == "fully_supported"
    ]
    cautions = [
        item for item in validate_cautions(output.comparison_cautions, registry)
        if item["evidence_status"] == "fully_supported"
    ]
    comparability = validate_comparability_assessments(
        output.comparability_assessment, paper_ids
    )
    common_limitations = [
        finding for finding in validate_findings(
            output.common_limitations, registry, ["limitations"]
        )
        if finding["evidence_status"] == "fully_supported"
        and len({item["paper_id"] for item in finding["evidence"]
                 if _is_valid_evidence(item)}) >= 2
    ]
    conflicts = [
        item for item in validate_conflicts(
            output.potential_conflicts, registry, comparability
        )
        if item["evidence_status"] == "fully_supported"
    ]
    return {
        "paper_ids": paper_ids,
        "comparison_table": table,
        "similarities": similarities,
        "differences": differences,
        "comparison_cautions": cautions,
        "comparability_assessment": comparability,
        "dataset_analysis": build_dataset_analysis(table, registry),
        "metric_analysis": build_metric_analysis(table, registry),
        "method_trends": build_method_trends(table, registry),
        "common_limitations": common_limitations,
        "potential_conflicts": conflicts,
        "missing_information": detect_missing_information(papers, registry),
        "evidence_coverage": calculate_evidence_coverage(
            similarities, differences, cautions, common_limitations, conflicts
        ),
        "overall_observations": output.overall_observations
    }

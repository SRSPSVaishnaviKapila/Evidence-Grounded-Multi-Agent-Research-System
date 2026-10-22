import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.services.llm_service import generate_structured_answer


ANALYSIS_FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations", "future_work"
)
VerificationStatus = Literal[
    "fully_supported", "partially_supported", "unsupported"
]
ClaimType = Literal[
    "paper_fact", "cross_paper_fact", "numeric_result", "limitation",
    "future_work", "candidate_gap", "generated_inference",
    "report_synthesis", "comparison_claim", "other"
]
ALLOWED_FIELD_GROUPS = {
    "objective": {"objective"},
    "methodology": {"methodology", "model"},
    "dataset": {"dataset", "methodology"},
    "model": {"model", "methodology"},
    "evaluation_metrics": {"evaluation_metrics", "main_results"},
    "main_results": {"main_results", "evaluation_metrics"},
    "limitations": {"limitations"},
    "future_work": {"future_work", "limitations"},
    "candidate_gap": {
        "limitations", "future_work", "methodology", "dataset", "model",
        "main_results", "evaluation_metrics"
    },
    "generated_inference": {
        "limitations", "future_work", "methodology", "dataset", "model",
        "main_results", "evaluation_metrics"
    },
    "report_synthesis": set(ANALYSIS_FIELDS),
    "comparison_claim": set(ANALYSIS_FIELDS)
}


class VerificationIssue(BaseModel):
    type: str
    severity: Literal["high", "medium", "low"]
    message: str


class CitationCheck(BaseModel):
    evidence_key_valid: bool
    paper_valid: bool
    chunk_valid: bool
    page_valid: bool
    field_valid: bool
    citation_valid: bool


class NumericCheck(BaseModel):
    claim_numbers: list[str] = Field(default_factory=list)
    supported_numbers: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)


class SemanticDecision(BaseModel):
    claim_id: str
    status: VerificationStatus
    issue_type: Optional[str] = None
    explanation: Optional[str] = None
    suggested_revision: Optional[str] = None


class SemanticVerificationOutput(BaseModel):
    decisions: list[SemanticDecision] = Field(default_factory=list)


def _value(item: Any, name: str):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _has_text(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def build_verification_evidence_registry(papers: list[Any]) -> dict[str, dict]:
    registry = {}
    for paper in papers:
        paper_id = int(_value(paper, "paper_id"))
        analysis = _value(paper, "analysis") or {}
        for field_name in ANALYSIS_FIELDS:
            field = _value(analysis, field_name) or {}
            registry[f"P{paper_id}_{field_name}"] = {
                "paper_id": paper_id,
                "field": field_name,
                "value": _value(field, "value"),
                "evidence_text": _value(field, "evidence_text"),
                "page_number": _value(field, "page_number"),
                "chunk_id": _value(field, "chunk_id"),
                "citation_valid": bool(_value(field, "citation_valid"))
            }
    return registry


def _evidence_ref(key: str, reference: dict) -> dict:
    return {
        "evidence_key": key,
        "paper_id": reference.get("paper_id"),
        "field": reference.get("field"),
        "page_number": reference.get("page_number"),
        "chunk_id": reference.get("chunk_id"),
        "citation_valid": reference.get("citation_valid")
    }


def _claim_type_for_field(field_name: str, value: str) -> str:
    if field_name == "limitations":
        return "limitation"
    if field_name == "future_work":
        return "future_work"
    if field_name == "main_results" and extract_numbers(value):
        return "numeric_result"
    return "paper_fact"


def collect_agent2_claims(
    papers: list[Any], registry: dict[str, dict]
) -> list[dict]:
    claims = []
    for paper in papers:
        paper_id = int(_value(paper, "paper_id"))
        for field_name in ANALYSIS_FIELDS:
            key = f"P{paper_id}_{field_name}"
            reference = registry[key]
            if not _has_text(reference["value"]):
                continue
            text = str(reference["value"]).strip()
            claims.append({
                "claim": text,
                "claim_type": _claim_type_for_field(field_name, text),
                "source_agent": "agent2",
                "source_section": field_name,
                "paper_ids": [paper_id],
                "expected_field": field_name,
                "evidence_refs": [_evidence_ref(key, reference)]
            })
    return claims


def _output_evidence(item: dict) -> list[dict]:
    return list(item.get("evidence") or [])


def collect_agent3_claims(output: Optional[dict]) -> list[dict]:
    if not output:
        return []
    claims = []
    mappings = {
        "similarities": ("statement", "comparison_claim"),
        "differences": ("statement", "comparison_claim"),
        "comparison_cautions": ("statement", "comparison_claim"),
        "common_limitations": ("statement", "limitation"),
        "potential_conflicts": ("description", "comparison_claim"),
        "method_trends": ("description", "cross_paper_fact")
    }
    for section, (text_field, claim_type) in mappings.items():
        for item in output.get(section, []) or []:
            text = item.get(text_field)
            evidence = _output_evidence(item)
            paper_ids = item.get("paper_ids") or sorted({
                ref.get("paper_id") for ref in evidence if ref.get("paper_id")
            })
            if _has_text(text):
                claims.append({
                    "claim": str(text).strip(), "claim_type": claim_type,
                    "source_agent": "agent3", "source_section": section,
                    "paper_ids": paper_ids,
                    "expected_field": (
                        "limitations" if section == "common_limitations"
                        else "comparison_claim"
                    ),
                    "evidence_refs": evidence
                })
    for observation in output.get("overall_observations", []) or []:
        if _has_text(observation):
            claims.append({
                "claim": str(observation).strip(),
                "claim_type": "cross_paper_fact",
                "source_agent": "agent3",
                "source_section": "overall_observations",
                "paper_ids": list(output.get("paper_ids") or []),
                "expected_field": "comparison_claim", "evidence_refs": []
            })
    return claims


REPORT_CITATION = re.compile(
    r"\[Paper\s+(\d+),\s*p\.([^,\]]+),\s*([^\]]+)\]",
    flags=re.IGNORECASE
)


def collect_report_claims(
    report: dict, registry: dict[str, dict]
) -> list[dict]:
    claims = []
    by_chunk = {
        (ref["paper_id"], str(ref["chunk_id"])): (key, ref)
        for key, ref in registry.items() if _has_text(ref.get("chunk_id"))
    }
    for section, content in report.items():
        if section == "title" or not _has_text(content):
            continue
        for paragraph in re.split(r"\n\s*\n", str(content)):
            citations = list(REPORT_CITATION.finditer(paragraph))
            text = REPORT_CITATION.sub("", paragraph).strip()
            if not text:
                continue
            refs, paper_ids = [], []
            for citation in citations:
                paper_id = int(citation.group(1))
                raw_page = citation.group(2).strip()
                page = int(raw_page) if raw_page.isdigit() else raw_page
                chunk_id = citation.group(3).strip()
                key_and_ref = by_chunk.get((paper_id, chunk_id))
                key = key_and_ref[0] if key_and_ref else f"P{paper_id}_unknown"
                refs.append({
                    "evidence_key": key, "paper_id": paper_id,
                    "page_number": page, "chunk_id": chunk_id
                })
                paper_ids.append(paper_id)
            claims.append({
                "claim": text, "claim_type": "report_synthesis",
                "source_agent": "agent4", "source_section": section,
                "paper_ids": list(dict.fromkeys(paper_ids)),
                "expected_field": "report_synthesis", "evidence_refs": refs
            })
    return claims


def collect_agent4_claims(
    output: Optional[dict], registry: dict[str, dict], verify_report: bool
) -> list[dict]:
    if not output:
        return []
    claims = []
    mappings = {
        "candidate_gaps": ("description", "candidate_gap", "candidate_gap"),
        "underexplored_areas": ("basis", "candidate_gap", "candidate_gap"),
        "future_direction_signals": (
            "direction", "generated_inference", "generated_inference"
        ),
        "research_questions": (
            "question", "generated_inference", "generated_inference"
        ),
        "research_opportunities": (
            "description", "generated_inference", "generated_inference"
        )
    }
    for section, (text_field, claim_type, expected_field) in mappings.items():
        for item in output.get(section, []) or []:
            text = item.get(text_field)
            if not _has_text(text):
                continue
            claims.append({
                "claim": str(text).strip(), "claim_type": claim_type,
                "source_agent": "agent4", "source_section": section,
                "paper_ids": list(
                    item.get("supporting_paper_ids")
                    or item.get("paper_ids") or []
                ),
                "expected_field": expected_field,
                "evidence_refs": _output_evidence(item)
            })
    if verify_report:
        claims.extend(collect_report_claims(output.get("report") or {}, registry))
    return claims


def extract_numbers(text: str) -> list[str]:
    clean = REPORT_CITATION.sub("", str(text))
    clean = re.sub(r"\bPaper\s+\d+\b", "", clean, flags=re.IGNORECASE)
    pattern = (
        r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?"
        r"(?:\s*(?:billion|million|thousand))?"
    )
    return list(dict.fromkeys(
        match.strip() for match in re.findall(pattern, clean, re.IGNORECASE)
    ))


def _normalized_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
        "with", "was", "were", "is", "are", "be", "by", "from", "that",
        "this", "these", "those", "paper", "study", "selected", "analyzed"
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if token not in stopwords and len(token) > 1
    }


def calculate_lexical_support(claim: str, evidence: list[dict]) -> float:
    claim_tokens = _normalized_tokens(claim)
    evidence_tokens = _normalized_tokens(" ".join(
        f"{item.get('value') or ''} {item.get('evidence_text') or ''}"
        for item in evidence
    ))
    if not claim_tokens:
        return 0.0
    return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 3)


def _issue(issue_type: str, severity: str, message: str) -> dict:
    return {"type": issue_type, "severity": severity, "message": message}


def validate_claim_deterministically(
    claim: dict, registry: dict[str, dict]
) -> dict:
    issues, evidence = [], []
    refs = claim.get("evidence_refs") or []
    expected_fields = ALLOWED_FIELD_GROUPS.get(
        claim.get("expected_field"), set(ANALYSIS_FIELDS)
    )
    checks = {
        "evidence_key_valid": bool(refs), "paper_valid": True,
        "chunk_valid": True, "page_valid": True, "field_valid": True,
        "citation_valid": True
    }
    for supplied in refs:
        key = supplied.get("evidence_key")
        trusted = registry.get(key)
        if trusted is None:
            checks["evidence_key_valid"] = False
            issues.append(_issue(
                "invalid_evidence_key", "high",
                f"Evidence key {key or '<missing>'} does not exist."
            ))
            continue
        if supplied.get("paper_id") not in (None, trusted["paper_id"]):
            checks["paper_valid"] = False
            issues.append(_issue(
                "citation_invalid", "high",
                f"Evidence key {key} belongs to Paper {trusted['paper_id']}."
            ))
        if supplied.get("chunk_id") not in (None, trusted["chunk_id"]):
            checks["chunk_valid"] = False
            issues.append(_issue(
                "chunk_mismatch", "high",
                f"Cited chunk does not match the registry for {key}."
            ))
        if supplied.get("page_number") not in (None, trusted["page_number"]):
            checks["page_valid"] = False
            issues.append(_issue(
                "page_mismatch", "high",
                f"Cited page does not match the registry for {key}."
            ))
        supplied_field = supplied.get("field")
        if supplied_field not in (None, trusted["field"]):
            checks["field_valid"] = False
        if expected_fields and trusted["field"] not in expected_fields:
            checks["field_valid"] = False
            issues.append(_issue(
                "field_mismatch", "medium",
                f"{trusted['field']} evidence is not appropriate for this claim."
            ))
        if not trusted.get("citation_valid"):
            checks["citation_valid"] = False
            issues.append(_issue(
                "citation_invalid", "high",
                f"Citation metadata for {key} is invalid."
            ))
        if _has_text(trusted.get("value")) and _has_text(trusted.get("evidence_text")):
            evidence.append({"evidence_key": key, **trusted})

    if not refs:
        issues.append(_issue(
            "missing_evidence", "high", "The claim has no cited evidence."
        ))

    required_papers = set(claim.get("paper_ids") or [])
    supported_papers = {item["paper_id"] for item in evidence}
    missing_papers = required_papers - supported_papers
    if missing_papers:
        issues.append(_issue(
            "missing_paper_support", "medium",
            "Missing valid evidence for paper(s): "
            + ", ".join(map(str, sorted(missing_papers))) + "."
        ))

    claim_numbers = extract_numbers(claim["claim"])
    evidence_numbers = set(extract_numbers(" ".join(
        f"{item.get('value') or ''} {item.get('evidence_text') or ''}"
        for item in evidence
    )))
    supported_numbers = [n for n in claim_numbers if n in evidence_numbers]
    unsupported_numbers = [n for n in claim_numbers if n not in evidence_numbers]
    if unsupported_numbers:
        issues.append(_issue(
            "numeric_mismatch", "high",
            "Numeric value(s) not supported by cited evidence: "
            + ", ".join(unsupported_numbers) + "."
        ))

    evidence_text = " ".join(
        f"{item.get('value') or ''} {item.get('evidence_text') or ''}"
        for item in evidence
    ).casefold()
    broad_terms = [
        term for term in ("all", "always", "proves", "superior", "never")
        if re.search(rf"\b{term}\b", claim["claim"], re.IGNORECASE)
        and not re.search(rf"\b{term}\b", evidence_text)
    ]
    if broad_terms:
        issues.append(_issue(
            "overstatement", "medium",
            "The claim is broader than the cited evidence."
        ))

    lexical_support = calculate_lexical_support(claim["claim"], evidence)
    hard_failure = (
        not evidence or not checks["evidence_key_valid"]
        or not checks["paper_valid"] or not checks["chunk_valid"]
        or not checks["page_valid"] or not checks["citation_valid"]
        or bool(unsupported_numbers)
    )
    if hard_failure or broad_terms:
        status = "unsupported"
        deterministic_cap = "unsupported"
    elif missing_papers or not checks["field_valid"]:
        status = "partially_supported" if evidence else "unsupported"
        deterministic_cap = status
    elif lexical_support >= (
        0.30 if claim.get("claim_type") in {
            "candidate_gap", "generated_inference"
        } else 0.45
    ):
        status = "fully_supported"
        deterministic_cap = "fully_supported"
    else:
        status = "partially_supported"
        deterministic_cap = "fully_supported"
        issues.append(_issue(
            "partial_support", "low",
            "Deterministic checks found limited lexical support; semantic review is needed."
        ))
    return {
        **{key: value for key, value in claim.items() if key != "evidence_refs"},
        "status": status,
        "evidence": [{
            key: item.get(key) for key in (
                "evidence_key", "paper_id", "field", "page_number",
                "chunk_id", "citation_valid"
            )
        } for item in evidence],
        "_semantic_evidence": evidence,
        "issues": issues,
        "citation_check": checks,
        "numeric_check": {
            "claim_numbers": claim_numbers,
            "supported_numbers": supported_numbers,
            "unsupported_numbers": unsupported_numbers
        },
        "lexical_support": lexical_support,
        "_hard_failure": hard_failure or bool(broad_terms),
        "_deterministic_cap": deterministic_cap
    }


def batch_semantic_verification(results: list[dict]) -> dict[str, dict]:
    candidates = [result for result in results if not result["_hard_failure"]]
    if not candidates:
        return {}
    context = [{
        "claim_id": item["claim_id"], "claim": item["claim"],
        "claim_type": item["claim_type"], "paper_ids": item["paper_ids"],
        "evidence": [{
            "paper_id": ref["paper_id"], "field": ref["field"],
            "value": ref["value"], "evidence_text": ref["evidence_text"]
        } for ref in item["_semantic_evidence"]]
    } for item in candidates]
    prompt = f"""
Audit every claim using ONLY its supplied evidence. Return fully_supported only
when all important details are present. Return partially_supported when the main
idea is supported but details are missing or broader. Return unsupported when the
evidence does not justify the claim. For candidate gaps and generated inferences,
judge whether the evidence reasonably motivates the inference rather than requiring
literal wording. Detect overstatement and semantic mismatch. Do not use outside
knowledge, correct claims from memory, invent facts, or treat missing evidence as
evidence of absence. Return one decision for every claim_id in one batch.

Claims:
{json.dumps(context, ensure_ascii=False, separators=(",", ":"))}
"""
    output = generate_structured_answer(
        prompt=prompt, response_schema=SemanticVerificationOutput
    )
    if not isinstance(output, SemanticVerificationOutput):
        output = SemanticVerificationOutput.model_validate(output)
    return {decision.claim_id: decision.model_dump() for decision in output.decisions}


def _combine_semantic(result: dict, decision: Optional[dict]) -> dict:
    if not decision or result["_hard_failure"]:
        return result
    rank = {"unsupported": 0, "partially_supported": 1, "fully_supported": 2}
    semantic_status = decision["status"]
    final_rank = min(
        rank[semantic_status], rank[result["_deterministic_cap"]]
    )
    result["status"] = next(
        status for status, value in rank.items() if value == final_rank
    )
    if result["status"] == "fully_supported":
        result["issues"] = [
            issue for issue in result["issues"]
            if issue["type"] != "partial_support"
        ]
    if semantic_status != "fully_supported":
        issue_type = decision.get("issue_type") or (
            "unsupported_claim" if semantic_status == "unsupported"
            else "partial_support"
        )
        if not any(issue["type"] == issue_type for issue in result["issues"]):
            result["issues"].append(_issue(
                issue_type,
                "medium" if semantic_status == "unsupported" else "low",
                decision.get("explanation") or "Semantic support is incomplete."
            ))
    if decision.get("suggested_revision"):
        result["suggested_revision"] = decision["suggested_revision"]
    return result


def build_verification_summary(results: list[dict]) -> dict:
    counts = Counter(item["status"] for item in results)
    total = len(results)
    return {
        "total_claims": total,
        "fully_supported": counts["fully_supported"],
        "partially_supported": counts["partially_supported"],
        "unsupported": counts["unsupported"],
        "support_ratio": round(
            counts["fully_supported"] / total if total else 0.0, 3
        )
    }


def build_agent_summary(results: list[dict]) -> dict:
    summary = {}
    for agent in ("agent2", "agent3", "agent4"):
        selected = [item for item in results if item["source_agent"] == agent]
        counts = Counter(item["status"] for item in selected)
        summary[agent] = {
            "total": len(selected),
            "fully_supported": counts["fully_supported"],
            "partially_supported": counts["partially_supported"],
            "unsupported": counts["unsupported"]
        }
    return summary


def build_critical_issues(results: list[dict]) -> list[dict]:
    critical = []
    for result in results:
        for issue in result["issues"]:
            if issue["severity"] == "high":
                critical.append({
                    "claim_id": result["claim_id"], "severity": "high",
                    "issue": issue["message"], "type": issue["type"]
                })
    return critical


def verify_evidence(
    papers: list[Any],
    agent3_output: Optional[dict] = None,
    agent4_output: Optional[dict] = None,
    verify_agent2: bool = True,
    verify_agent3: bool = True,
    verify_agent4: bool = True,
    verify_report: bool = True,
    warnings: Optional[list[str]] = None
) -> dict:
    registry = build_verification_evidence_registry(papers)
    claims = []
    if verify_agent2:
        claims.extend(collect_agent2_claims(papers, registry))
    if verify_agent3:
        claims.extend(collect_agent3_claims(agent3_output))
    if verify_agent4:
        claims.extend(collect_agent4_claims(
            agent4_output, registry, verify_report
        ))
    for index, claim in enumerate(claims, 1):
        claim["claim_id"] = f"CLM_{index:03d}"
    results = [
        validate_claim_deterministically(claim, registry) for claim in claims
    ]
    output_warnings = list(warnings or [])
    try:
        decisions = batch_semantic_verification(results)
    except Exception:
        decisions = {}
        output_warnings.append(
            "Semantic verification unavailable; deterministic verification was completed."
        )
    results = [_combine_semantic(item, decisions.get(item["claim_id"]))
               for item in results]
    for item in results:
        item.pop("_semantic_evidence", None)
        item.pop("_hard_failure", None)
        item.pop("_deterministic_cap", None)
        item.pop("expected_field", None)
    summary = build_verification_summary(results)
    critical = build_critical_issues(results)
    if (
        summary["support_ratio"] >= 0.9
        and not critical
    ):
        quality = "high"
    elif summary["support_ratio"] >= 0.7 and len(critical) <= 1:
        quality = "medium"
    else:
        quality = "low"
    return {
        "paper_ids": [int(_value(paper, "paper_id")) for paper in papers],
        "claim_results": results,
        "summary": summary,
        "agent_summary": build_agent_summary(results),
        "critical_issues": critical,
        "verification_quality": quality,
        "warnings": list(dict.fromkeys(output_warnings))
    }

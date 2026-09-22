from typing import Optional

from pydantic import BaseModel

from app.services.vector_store_service import (
    search_chunks
)

from app.services.llm_service import (
    generate_structured_answer
)


# =========================================================
# STRUCTURED OUTPUT SCHEMAS
# =========================================================

class EvidenceField(BaseModel):
    value: Optional[str] = None

    evidence_text: Optional[str] = None

    page_number: Optional[int] = None

    chunk_id: Optional[str] = None


class PaperAnalysisOutput(BaseModel):

    objective: EvidenceField

    methodology: EvidenceField

    dataset: EvidenceField

    model: EvidenceField

    evaluation_metrics: EvidenceField

    main_results: EvidenceField

    limitations: EvidenceField

    future_work: EvidenceField


# =========================================================
# RETRIEVAL QUERIES
# =========================================================

FIELD_QUERIES = {

    "objective": [
        "overall objective purpose goal contribution of the paper",
        "main aim proposed system research objective",
        "what problem does this work address",
        "to overcome these challenges we constructed trained model "
        "systematically demonstrated performance and real-world utility"
    ],
    "methodology": [
        "overall methodology training framework proposed approach",
        "model training pipeline architecture pretraining fine-tuning",
        "image-text learning training strategy experimental framework"
    ],
    "dataset": [
        "datasets used for model training validation and evaluation",
        "data sources cohorts samples training and test datasets",
        "final training dataset data samples collected source datasets "
        "unique images and annotations"
    ],
    "model": [
        "proposed model architecture and model components",
        "image encoder text encoder language model architecture",
        "vision-language foundation CNN transformer multimodal architecture"
    ],
    "evaluation_metrics": [
        "evaluation metrics performance measures primary endpoints",
        "reported quantitative metrics evaluation criteria"
    ],
    "main_results": [
        "main experimental results findings and conclusions",
        "primary reported performance outcomes"
    ],
    "limitations": [
        "limitations weaknesses challenges constraints discussion",
        "generalizability clinical deployment external validation limitations",
        "dataset bias failure cases and stated limitations"
    ],
    "future_work": [
        "explicit future work and future directions",
        "further research next steps future improvements",
        "remaining challenges proposed for future study",
        "model scaling autonomous systems self-improvement future work",
        "future studies expand evaluation scope education and deployment"
    ]
}


SECTION_PREFERENCES = {
    "objective": (
        "abstract", "introduction", "overview", "contribution"
    ),
    "methodology": (
        "methods", "methodology", "approach", "training",
        "pretraining", "framework"
    ),
    "model": (
        "model", "architecture", "framework", "image encoder",
        "text encoder", "language model"
    ),
    "limitations": (
        "discussion", "limitation", "generalizability",
        "external validation"
    ),
    "future_work": (
        "future work", "future direction", "further research",
        "conclusion"
    )
}


PAPER_LEVEL_SIGNALS = (
    "this paper", "this work", "we propose", "we present",
    "we constructed", "we developed", "we systematically demonstrated",
    "our approach", "our framework", "overall", "main contribution"
)


SUB_EXPERIMENT_SIGNALS = (
    "reader study", "ablation", "editing time", "benchmark experiment",
    "case study", "single experiment"
)


def rank_field_candidates(
    field_name: str,
    candidates: list[dict],
    limit: int = 6
) -> list[dict]:
    """Apply soft section and paper-level preferences to retrieval."""

    ranked = []

    for candidate in candidates:
        text = candidate.get("text", "").casefold()
        score = float(candidate.get("distance", 1.0) or 0.0)

        for signal in SECTION_PREFERENCES.get(field_name, ()):
            if signal in text:
                score -= 0.08

        if field_name in ("objective", "methodology", "model"):
            if any(signal in text for signal in PAPER_LEVEL_SIGNALS):
                score -= 0.12
            if any(signal in text for signal in SUB_EXPERIMENT_SIGNALS):
                score += 0.18

        if field_name == "limitations":
            if "limitation" in text or "generalizability" in text:
                score -= 0.12
            if "failure example" in text or "failure case" in text:
                score += 0.08

        if field_name == "future_work":
            if any(signal in text for signal in (
                "future work", "future direction", "further research",
                "future study", "next step"
            )):
                score -= 0.12

        ranked.append((score, candidate))

    ranked.sort(key=lambda item: (
        item[0],
        item[1].get("page_number", 0),
        item[1].get("chunk_id", "")
    ))
    return [candidate for _, candidate in ranked[:limit]]


# =========================================================
# RETRIEVE EVIDENCE
# =========================================================

def retrieve_analysis_evidence(
    paper_id: int,
    top_k_per_field: int = 3
):
    """
    Run separate semantic searches for each
    analysis field.

    Duplicate chunks are removed.
    """

    unique_chunks = {}

    candidate_limit = min(
        6,
        max(3, top_k_per_field * 2)
    )

    for field_name, queries in FIELD_QUERIES.items():
        field_candidates = {}

        for query in queries:
            results = search_chunks(
                query=query,
                paper_id=paper_id,
                # Search a slightly wider local pool, then retain only
                # the best 3-6 field-specific candidates below.
                n_results=max(
                    10 if field_name == "objective" else 5,
                    top_k_per_field
                )
            )

            for chunk in results:
                chunk_id = chunk["chunk_id"]
                existing = field_candidates.get(chunk_id)
                if (
                    existing is None
                    or chunk.get("distance", 1.0)
                    < existing.get("distance", 1.0)
                ):
                    field_candidates[chunk_id] = chunk

        selected_chunks = rank_field_candidates(
            field_name,
            list(field_candidates.values()),
            candidate_limit
        )

        for chunk in selected_chunks:

            chunk_id = chunk["chunk_id"]

            if chunk_id not in unique_chunks:

                unique_chunks[chunk_id] = {
                    **chunk,
                    "retrieved_for": [
                        field_name
                    ]
                }

            else:

                if field_name not in unique_chunks[
                    chunk_id
                ]["retrieved_for"]:
                    unique_chunks[
                        chunk_id
                    ]["retrieved_for"].append(
                        field_name
                    )

    return list(
        unique_chunks.values()
    )


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_analysis_context(
    chunks: list[dict]
):
    """
    Create evidence context for Gemini.
    """

    context_parts = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        retrieved_for = ", ".join(
            chunk["retrieved_for"]
        )

        part = f"""
Evidence {index}

Chunk ID: {chunk["chunk_id"]}
Paper ID: {chunk["paper_id"]}
Page Number: {chunk["page_number"]}
Retrieved For: {retrieved_for}

Text:
{chunk["text"]}
"""

        context_parts.append(
            part.strip()
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# CITATION VALIDATION
# =========================================================

def validate_analysis_citations(
    analysis: PaperAnalysisOutput,
    retrieved_chunks: list[dict]
):
    """
    Verify that Gemini only cites chunk IDs
    that were actually retrieved.

    Returns normal dictionary with an extra
    citation_valid field for each analysis item.
    """

    valid_chunks = {
        chunk["chunk_id"]: chunk
        for chunk in retrieved_chunks
    }

    analysis_dict = analysis.model_dump()

    for field_name, field_data in (
        analysis_dict.items()
    ):

        chunk_id = field_data.get(
            "chunk_id"
        )

        page_number = field_data.get(
            "page_number"
        )

        citation_valid = False

        if chunk_id in valid_chunks:

            actual_page = valid_chunks[
                chunk_id
            ]["page_number"]

            retrieved_for = valid_chunks[
                chunk_id
            ].get("retrieved_for", [])

            if (
                actual_page
                == page_number
                and field_name in retrieved_for
            ):
                citation_valid = True

        field_data[
            "citation_valid"
        ] = citation_valid

    return analysis_dict


# =========================================================
# MAIN PAPER ANALYSIS AGENT
# =========================================================

def analyze_paper(
    paper_id: int,
    top_k_per_field: int = 3
):
    """
    Complete Agent 2 pipeline.
    """

    # -----------------------------------------
    # STEP 1: Retrieve evidence
    # -----------------------------------------

    retrieved_chunks = (
        retrieve_analysis_evidence(
            paper_id=paper_id,
            top_k_per_field=top_k_per_field
        )
    )

    if not retrieved_chunks:

        raise ValueError(
            "No evidence was retrieved "
            "for this paper."
        )

    # -----------------------------------------
    # STEP 2: Build context
    # -----------------------------------------

    context = build_analysis_context(
        retrieved_chunks
    )

    # -----------------------------------------
    # STEP 3: Prompt
    # -----------------------------------------

    prompt = f"""
You are the Paper Analysis and Evidence Retrieval
Agent in an evidence-grounded research system.

Analyze the research paper using ONLY the evidence
provided below.

Extract the following information:

- Main research objective
- Methodology
- Dataset
- Model or architecture
- Evaluation metrics
- Main results
- Limitations
- Future work

IMPORTANT RULES:

1. Use only the retrieved evidence provided below.

2. Do not use outside knowledge.

3. Do not invent missing information.

4. Every extracted field must cite the single
   strongest supporting chunk.

5. The chunk ID must exactly match one of the
   retrieved chunk IDs.

6. The page number must exactly match the page
   number belonging to that chunk.

7. evidence_text should contain a short supporting
   passage or close evidence-based summary from
   that chunk.

8. If a field cannot be determined from the
   retrieved evidence, return null values for that
   field instead of guessing.

9. For datasets, models, metrics, results,
   limitations, or future work containing multiple
   items, combine the supported items clearly in
   the value field.

10. A field may cite only a chunk whose "Retrieved For"
    list contains that exact field name.

11. For objective, extract the main paper-level purpose,
    problem, or contribution. Prefer abstract,
    introduction, overview, and contribution evidence.

12. For methodology, extract the core technical approach,
    training strategy, and development framework. Do not
    let preprocessing or one evaluation procedure replace
    the overall methodology.

13. For model, describe the supported architecture, type,
    and major components sufficiently for cross-paper
    comparison. Do not reduce a broader model to one task
    it performs.

14. For objective, methodology, and model, prefer evidence
    describing the overall paper or system over a reader
    study, ablation, benchmark, or other sub-experiment.
    Use sub-experiment evidence as the main answer only
    when no broader supported evidence exists.

15. For limitations, prefer explicit author-stated,
    paper-level limitations over an isolated failure
    example. Do not turn a metric value into a limitation.

16. For future work, return a value only when the evidence
    explicitly states a future direction, next step, or
    proposed future improvement. Otherwise return null.

17. Always select the strongest matching chunk ID and its
    exact page number. If evidence for a field is indirect,
    narrow, or insufficient, return null rather than infer.

Retrieved Evidence:

{context}
"""

    # -----------------------------------------
    # STEP 4: Structured LLM analysis
    # -----------------------------------------

    analysis = generate_structured_answer(
        prompt=prompt,
        response_schema=PaperAnalysisOutput
    )

    # -----------------------------------------
    # STEP 5: Validate citations
    # -----------------------------------------

    validated_analysis = (
        validate_analysis_citations(
            analysis=analysis,
            retrieved_chunks=retrieved_chunks
        )
    )

    return {
        "paper_id": paper_id,

        "analysis":
            validated_analysis,

        "retrieved_chunks":
            len(retrieved_chunks)
    }

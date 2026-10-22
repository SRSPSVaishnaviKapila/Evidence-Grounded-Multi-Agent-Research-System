import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agents.cross_paper_agent import compare_papers
from app.agents.evidence_verification_agent import verify_evidence
from app.agents.research_gap_report_agent import analyze_research_gaps
from app.database.database import get_db
from app.database.models import Paper, PaperAnalysis
from app.database.schemas import VerificationRequest


router = APIRouter(
    prefix="/verification",
    tags=["Evidence Verification"]
)
logger = logging.getLogger(__name__)
ANALYSIS_FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations", "future_work"
)


@router.post("/check")
def check_evidence(
    request: VerificationRequest,
    db: Session = Depends(get_db)
):
    paper_ids = request.paper_ids
    papers_by_id = {
        paper.id: paper for paper in db.query(Paper).filter(
            Paper.id.in_(paper_ids)
        ).all()
    }
    missing_ids = [paper_id for paper_id in paper_ids
                   if paper_id not in papers_by_id]
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail="Paper(s) not found: " + ", ".join(map(str, missing_ids))
        )
    analyses_by_id = {
        analysis.paper_id: analysis
        for analysis in db.query(PaperAnalysis).filter(
            PaperAnalysis.paper_id.in_(paper_ids)
        ).all()
    }
    unanalyzed_ids = [paper_id for paper_id in paper_ids
                      if paper_id not in analyses_by_id]
    if unanalyzed_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "Paper(s) must be analyzed first: "
                + ", ".join(map(str, unanalyzed_ids))
            )
        )

    agent_input = []
    for paper_id in paper_ids:
        paper, analysis = papers_by_id[paper_id], analyses_by_id[paper_id]
        agent_input.append({
            "paper_id": paper_id, "title": paper.title, "year": paper.year,
            "analysis": {
                field_name: getattr(analysis, field_name)
                for field_name in ANALYSIS_FIELDS
            }
        })

    warnings, agent3_output, agent4_output = [], None, None
    requires_cross_paper = request.verify_agent3 or request.verify_agent4
    if requires_cross_paper and len(agent_input) < 2:
        warnings.append(
            "Agent 3 and Agent 4 verification require at least two papers; "
            "Agent 2 verification was completed."
        )
    elif requires_cross_paper:
        try:
            agent3_output = compare_papers(agent_input)
        except Exception:
            logger.exception("Agent 3 output unavailable during verification")
            warnings.append(
                "Agent 3 output unavailable; its claims were not verified."
            )
        if request.verify_agent4 and agent3_output is not None:
            try:
                agent4_output = analyze_research_gaps(
                    agent_input, agent3_output
                )
            except Exception:
                logger.exception("Agent 4 output unavailable during verification")
                warnings.append(
                    "Agent 4 output unavailable; its claims were not verified."
                )
        elif request.verify_agent4:
            warnings.append(
                "Agent 4 output unavailable because Agent 3 output was not "
                "available; its claims were not verified."
            )

    result = verify_evidence(
        papers=agent_input,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
        verify_agent2=request.verify_agent2,
        verify_agent3=request.verify_agent3,
        verify_agent4=request.verify_agent4,
        verify_report=request.verify_report,
        warnings=warnings
    )
    return {
        **result,
        "status": "Evidence verification completed successfully"
    }

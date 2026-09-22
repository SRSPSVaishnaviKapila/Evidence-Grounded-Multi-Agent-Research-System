import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from google.genai import errors as genai_errors
from sqlalchemy.orm import Session

from app.agents.cross_paper_agent import compare_papers
from app.agents.research_gap_report_agent import analyze_research_gaps
from app.database.database import get_db
from app.database.models import Paper, PaperAnalysis
from app.database.schemas import ResearchGapRequest


router = APIRouter(
    prefix="/gaps",
    tags=["Research Gap & Report"]
)

logger = logging.getLogger(__name__)


@router.post("/analyze")
def analyze_gaps(
    request: ResearchGapRequest,
    db: Session = Depends(get_db)
):
    paper_ids = list(dict.fromkeys(request.paper_ids))
    if len(paper_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two unique paper IDs are required."
        )

    papers_by_id = {
        paper.id: paper
        for paper in db.query(Paper).filter(
            Paper.id.in_(paper_ids)
        ).all()
    }
    missing_ids = [
        paper_id for paper_id in paper_ids
        if paper_id not in papers_by_id
    ]
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=("Paper(s) not found: " + ", ".join(map(str, missing_ids)))
        )

    analyses_by_id = {
        analysis.paper_id: analysis
        for analysis in db.query(PaperAnalysis).filter(
            PaperAnalysis.paper_id.in_(paper_ids)
        ).all()
    }
    unanalyzed_ids = [
        paper_id for paper_id in paper_ids
        if paper_id not in analyses_by_id
    ]
    if unanalyzed_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "Paper(s) must be analyzed first: "
                + ", ".join(map(str, unanalyzed_ids))
                + ". Run POST /papers/{paper_id}/analyze."
            )
        )

    agent_input = []
    for paper_id in paper_ids:
        paper = papers_by_id[paper_id]
        analysis = analyses_by_id[paper_id]
        agent_input.append({
            "paper_id": paper_id,
            "title": paper.title,
            "year": paper.year,
            "analysis": {
                field_name: getattr(analysis, field_name)
                for field_name in (
                    "objective", "methodology", "dataset", "model",
                    "evaluation_metrics", "main_results", "limitations",
                    "future_work"
                )
            }
        })

    try:
        cross_paper_output = compare_papers(agent_input)
        result = analyze_research_gaps(agent_input, cross_paper_output)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except genai_errors.ClientError as error:
        if (
            getattr(error, "code", None) == 429
            or "RESOURCE_EXHAUSTED" in str(error)
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. Wait for the quota to reset "
                    "or use a project with available Gemini quota, then retry."
                )
            ) from error
        logger.exception("Gemini rejected the Agent 4 request")
        raise HTTPException(
            status_code=502,
            detail="Gemini rejected the research gap analysis request."
        ) from error
    except RuntimeError as error:
        logger.exception("Gemini service failed during Agent 4 analysis")
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily unavailable. Please retry later."
        ) from error
    except httpx.HTTPError as error:
        logger.exception("Gemini connection failed during Agent 4 analysis")
        raise HTTPException(
            status_code=503,
            detail="Gemini is temporarily unavailable. Please retry later."
        ) from error
    except Exception as error:
        logger.exception("Unexpected Agent 4 failure")
        raise HTTPException(
            status_code=500,
            detail="Research gap analysis failed. Please try again."
        ) from error

    return {
        **result,
        "status": "Research gap analysis completed successfully"
    }

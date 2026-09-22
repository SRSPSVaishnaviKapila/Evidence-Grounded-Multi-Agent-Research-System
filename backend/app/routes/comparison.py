from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agents.cross_paper_agent import compare_papers
from app.database.database import get_db
from app.database.models import Paper, PaperAnalysis
from app.database.schemas import CrossPaperRequest


router = APIRouter(
    prefix="/comparison",
    tags=["Cross-Paper Analysis"]
)


@router.post("/")
def compare_analyzed_papers(
    request: CrossPaperRequest,
    db: Session = Depends(get_db)
):
    paper_ids = list(dict.fromkeys(request.paper_ids))

    if len(paper_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two unique paper IDs "
                "are required."
            )
        )

    papers = {
        paper.id: paper
        for paper in db.query(Paper).filter(
            Paper.id.in_(paper_ids)
        ).all()
    }

    missing_ids = [
        paper_id
        for paper_id in paper_ids
        if paper_id not in papers
    ]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                "Paper(s) not found: "
                + ", ".join(map(str, missing_ids))
            )
        )

    analyses = {
        analysis.paper_id: analysis
        for analysis in db.query(PaperAnalysis).filter(
            PaperAnalysis.paper_id.in_(paper_ids)
        ).all()
    }

    unanalyzed_ids = [
        paper_id
        for paper_id in paper_ids
        if paper_id not in analyses
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
        paper = papers[paper_id]
        analysis = analyses[paper_id]
        agent_input.append({
            "paper_id": paper_id,
            "title": paper.title,
            "year": paper.year,
            "analysis": {
                field_name: getattr(
                    analysis,
                    field_name
                )
                for field_name in (
                    "objective",
                    "methodology",
                    "dataset",
                    "model",
                    "evaluation_metrics",
                    "main_results",
                    "limitations",
                    "future_work"
                )
            }
        })

    try:
        result = compare_papers(agent_input)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cross-paper analysis failed. "
                "Please try again."
            )
        ) from error

    return {
        **result,
        "total_papers": len(paper_ids),
        "status": (
            "Cross-paper analysis completed "
            "successfully"
        )
    }

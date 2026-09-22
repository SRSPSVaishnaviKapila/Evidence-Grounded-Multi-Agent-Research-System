import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.agents.literature_search_agent import (
    LiteratureSearchError,
    search_literature
)
from app.database.database import get_db
from app.database.models import Paper
from app.database.schemas import (
    LiteratureImportRequest,
    LiteratureSearchRequest
)


router = APIRouter(
    prefix="/literature",
    tags=["Literature Search"]
)


@router.post("/search")
def search(request: LiteratureSearchRequest):
    try:
        results = search_literature(
            query=request.query,
            limit=request.limit,
            from_year=request.from_year,
            to_year=request.to_year
        )
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Literature search provider timed out."
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail="Literature search provider is unavailable."
        ) from error
    except LiteratureSearchError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "query": request.query,
        "count": len(results),
        "results": results
    }


def _existing_paper(db: Session, doi: str | None, title: str, year: int | None):
    if doi:
        existing = db.query(Paper).filter(
            func.lower(Paper.doi) == doi.casefold()
        ).first()
        if existing:
            return existing
    query = db.query(Paper).filter(
        func.lower(Paper.title) == title.casefold()
    )
    if year is not None:
        query = query.filter(Paper.year == year)
    return query.first()


@router.post("/import")
def import_papers(
    request: LiteratureImportRequest,
    db: Session = Depends(get_db)
):
    imported, skipped = [], []
    for item in request.papers:
        existing = _existing_paper(db, item.doi, item.title, item.year)
        if existing:
            skipped.append({
                "result_id": item.result_id,
                "paper_id": existing.id,
                "reason": "already_exists"
            })
            continue
        paper = Paper(
            title=item.title.strip(),
            authors=", ".join(item.authors) if item.authors else None,
            year=item.year,
            abstract=item.abstract,
            doi=item.doi,
            url=item.url,
            source=item.source
        )
        db.add(paper)
        db.flush()
        imported.append({
            "result_id": item.result_id,
            "paper_id": paper.id,
            "title": paper.title
        })
    db.commit()
    return {
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported": imported,
        "skipped": skipped
    }

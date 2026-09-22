from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    HTTPException
)
import traceback
from sqlalchemy.orm import Session
from pathlib import Path

import shutil
import pymupdf

from app.database.database import get_db
from app.database.models import (Paper, PaperAnalysis)

from app.database.schemas import (
    PaperCreate,
    PaperResponse,
    PDFUploadResponse,
    RAGQuestion
)
from app.services.rag_service import (
    answer_question
)

from app.services.pdf_service import (
    extract_text_from_pdf,
    save_extracted_text
)

from app.services.chunking_service import (
    create_chunks
)

from app.services.vector_store_service import (
    store_chunks,
    search_chunks,
    get_collection_count,
    get_paper_vector_count
)
from app.agents.paper_analysis_agent import (
    analyze_paper
)


router = APIRouter(
    prefix="/papers",
    tags=["Papers"]
)



# =========================================================
# CREATE PAPER
# =========================================================

@router.post(
    "/",
    response_model=PaperResponse
)
def create_paper(
    paper: PaperCreate,
    db: Session = Depends(get_db)
):

    new_paper = Paper(
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=paper.abstract,
        doi=paper.doi,
        url=paper.url,
        source=paper.source
    )

    db.add(new_paper)
    db.commit()
    db.refresh(new_paper)

    return new_paper


# =========================================================
# GET ALL PAPERS
# =========================================================

@router.get(
    "/",
    response_model=list[PaperResponse]
)
def get_papers(
    db: Session = Depends(get_db)
):

    papers = db.query(Paper).all()

    return papers


# =========================================================
# UPLOAD PDF
# =========================================================

@router.post(
    "/{paper_id}/upload",
    response_model=PDFUploadResponse
)
async def upload_paper_pdf(
    paper_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    pdf_directory = Path(
        "data/papers"
    )

    text_directory = Path(
        "data/extracted_text"
    )

    pdf_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    text_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_filename = (
        f"paper_{paper_id}.pdf"
    )

    pdf_path = (
        pdf_directory /
        pdf_filename
    )

    # Save uploaded PDF
    with open(
        pdf_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        # Extract text
        extracted_text = (
            extract_text_from_pdf(
                str(pdf_path)
            )
        )

        # Count pages
        document = pymupdf.open(
            str(pdf_path)
        )

        pages_processed = len(
            document
        )

        document.close()

        # Save extracted text
        text_filename = (
            f"paper_{paper_id}.txt"
        )

        text_path = (
            text_directory /
            text_filename
        )

        save_extracted_text(
            extracted_text,
            str(text_path)
        )

        # Update database
        paper.pdf_path = str(
            pdf_path
        )

        paper.text_path = str(
            text_path
        )

        db.commit()
        db.refresh(paper)

        return {
            "paper_id":
                paper.id,

            "filename":
                file.filename,

            "pdf_path":
                str(pdf_path),

            "text_path":
                str(text_path),

            "pages_processed":
                pages_processed,

            "status":
                "PDF uploaded and text extracted successfully"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed: "
                f"{str(error)}"
            )
        )


# =========================================================
# GET PAPER CHUNKS
# =========================================================

@router.get(
    "/{paper_id}/chunks"
)
def get_paper_chunks(
    paper_id: int,
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    if not paper.text_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "PDF text has not "
                "been extracted yet"
            )
        )

    try:

        with open(
            paper.text_path,
            "r",
            encoding="utf-8"
        ) as file:

            extracted_text = (
                file.read()
            )

        chunks = create_chunks(
            extracted_text,
            paper_id
        )

        return {
            "paper_id":
                paper_id,

            "total_chunks":
                len(chunks),

            "chunks":
                chunks
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Chunking failed: "
                f"{str(error)}"
            )
        )


# =========================================================
# INDEX PAPER INTO CHROMADB
# =========================================================

@router.post(
    "/{paper_id}/index"
)
def index_paper(
    paper_id: int,
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    if not paper.text_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Paper text has not "
                "been extracted"
            )
        )

    try:

        with open(
            paper.text_path,
            "r",
            encoding="utf-8"
        ) as file:

            extracted_text = (
                file.read()
            )

        chunks = create_chunks(
            extracted_text,
            paper_id
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No chunks were "
                    "generated"
                )
            )

        total_stored = store_chunks(
            chunks
        )

        return {
            "paper_id":
                paper_id,

            "total_chunks":
                len(chunks),

            "stored_chunks":
                total_stored,

            "paper_vectors":
                get_paper_vector_count(
                    paper_id
                ),

            "vector_database_count":
                get_collection_count(),

            "status":
                "Paper indexed successfully"
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Indexing failed: "
                f"{str(error)}"
            )
        )


# =========================================================
# VECTOR STATUS
# =========================================================

@router.get(
    "/{paper_id}/vector-status"
)
def vector_status(
    paper_id: int,
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:

        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    return {
        "paper_id":
            paper_id,

        "paper_vectors":
            get_paper_vector_count(
                paper_id
            ),

        "total_vectors":
            get_collection_count()
    }


# =========================================================
# SEMANTIC SEARCH
# =========================================================

@router.get(
    "/{paper_id}/search"
)
def search_paper(
    paper_id: int,
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:

        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    if limit < 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "Limit must be "
                "greater than 0"
            )
        )

    # Important debugging check
    paper_vectors = (
        get_paper_vector_count(
            paper_id
        )
    )

    if paper_vectors == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "This paper has not "
                "been indexed yet. "
                "Run POST "
                f"/papers/{paper_id}/index first."
            )
        )

    try:

        results = search_chunks(
            query=query,
            paper_id=paper_id,
            n_results=limit
        )

        return {
            "paper_id":
                paper_id,

            "query":
                query,

            "paper_vectors":
                paper_vectors,

            "total_results":
                len(results),

            "results":
                results
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Search failed: "
                f"{str(error)}"
            )
        )
# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

@router.post(
    "/{paper_id}/ask"
)
def ask_paper(
    paper_id: int,
    request: RAGQuestion,
    db: Session = Depends(get_db)
):

    # Check paper exists
    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:

        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    # Check vector indexing
    paper_vectors = (
        get_paper_vector_count(
            paper_id
        )
    )

    if paper_vectors == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Paper has not been indexed. "
                f"Run POST /papers/{paper_id}/index first."
            )
        )

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if (
        request.top_k < 1
        or request.top_k > 10
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "top_k must be between "
                "1 and 10"
            )
        )

    try:

        result = answer_question(
            paper_id=paper_id,
            question=request.question,
            top_k=request.top_k
        )

        return {
            "paper_id":
                paper_id,

            "question":
                request.question,

            "answer":
                result["answer"],

            "evidence":
                result["evidence"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG processing failed: "
                f"{str(error)}"
            )
        )
# =========================================================
# PAPER ANALYSIS AGENT
# =========================================================

@router.post(
    "/{paper_id}/analyze"
)
def analyze_research_paper(
    paper_id: int,
    db: Session = Depends(get_db)
):

    # Check paper
    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:

        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    # Check vector indexing
    paper_vectors = (
        get_paper_vector_count(
            paper_id
        )
    )

    if paper_vectors == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Paper has not been indexed. "
                f"Run POST /papers/{paper_id}/index first."
            )
        )

    try:

        # Run Agent 2
        result = analyze_paper(
            paper_id=paper_id,
            top_k_per_field=3
        )

        analysis = result[
            "analysis"
        ]

        # Check whether analysis already exists
        existing_analysis = (
            db.query(PaperAnalysis)
            .filter(
                PaperAnalysis.paper_id
                == paper_id
            )
            .first()
        )

        # -----------------------------------
        # UPDATE
        # -----------------------------------

        if existing_analysis:

            existing_analysis.objective = (
                analysis["objective"]
            )

            existing_analysis.methodology = (
                analysis["methodology"]
            )

            existing_analysis.dataset = (
                analysis["dataset"]
            )

            existing_analysis.model = (
                analysis["model"]
            )

            existing_analysis.evaluation_metrics = (
                analysis[
                    "evaluation_metrics"
                ]
            )

            existing_analysis.main_results = (
                analysis["main_results"]
            )

            existing_analysis.limitations = (
                analysis["limitations"]
            )

            existing_analysis.future_work = (
                analysis["future_work"]
            )

            analysis_record = (
                existing_analysis
            )

        # -----------------------------------
        # CREATE
        # -----------------------------------

        else:

            analysis_record = PaperAnalysis(
                paper_id=paper_id,

                objective=
                    analysis["objective"],

                methodology=
                    analysis["methodology"],

                dataset=
                    analysis["dataset"],

                model=
                    analysis["model"],

                evaluation_metrics=
                    analysis[
                        "evaluation_metrics"
                    ],

                main_results=
                    analysis["main_results"],

                limitations=
                    analysis["limitations"],

                future_work=
                    analysis["future_work"]
            )

            db.add(
                analysis_record
            )

        db.commit()

        db.refresh(
            analysis_record
        )

        return {
            "paper_id":
                paper_id,

            "title":
                paper.title,

            "analysis_id":
                analysis_record.id,

            "analysis":
                analysis,

            "retrieved_chunks":
                result[
                    "retrieved_chunks"
                ],

            "status":
                "Paper analyzed and saved successfully"
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Paper analysis failed: "
                f"{str(error)}"
            )
        )
@router.get(
    "/{paper_id}/analysis"
)
def get_paper_analysis(
    paper_id: int,
    db: Session = Depends(get_db)
):

    paper = db.query(Paper).filter(
        Paper.id == paper_id
    ).first()

    if not paper:

        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )

    analysis = (
        db.query(PaperAnalysis)
        .filter(
            PaperAnalysis.paper_id
            == paper_id
        )
        .first()
    )

    if not analysis:

        raise HTTPException(
            status_code=404,
            detail=(
                "Paper has not been "
                "analyzed yet"
            )
        )

    return {
        "paper_id":
            paper_id,

        "title":
            paper.title,

        "analysis": {

            "objective":
                analysis.objective,

            "methodology":
                analysis.methodology,

            "dataset":
                analysis.dataset,

            "model":
                analysis.model,

            "evaluation_metrics":
                analysis.evaluation_metrics,

            "main_results":
                analysis.main_results,

            "limitations":
                analysis.limitations,

            "future_work":
                analysis.future_work
        },

        "created_at":
            analysis.created_at,

        "updated_at":
            analysis.updated_at
    }
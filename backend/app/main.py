from fastapi import FastAPI

from app.database.database import engine, Base
from app.routes import comparison, gaps, literature, papers, verification


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Evidence-Grounded Multi-Agent Research System",
    description="Backend API for automated literature review and cross-paper research analysis.",
    version="1.0.0"
)


app.include_router(papers.router)
app.include_router(comparison.router)
app.include_router(gaps.router)
app.include_router(literature.router)
app.include_router(verification.router)


@app.get("/")
def root():
    return {
        "message": "Evidence-Grounded Research System Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

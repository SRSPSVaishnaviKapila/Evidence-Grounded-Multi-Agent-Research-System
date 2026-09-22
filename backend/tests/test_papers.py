import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.database import Base
from app.database.models import Paper, PaperAnalysis
from app.routes.papers import analyze_research_paper


FIELDS = (
    "objective", "methodology", "dataset", "model",
    "evaluation_metrics", "main_results", "limitations", "future_work"
)


def analysis_result(version):
    return {
        "paper_id": 1,
        "retrieved_chunks": 8,
        "analysis": {
            field: {
                "value": f"{version} {field}",
                "evidence_text": f"Evidence {field}",
                "page_number": 1,
                "chunk_id": "P1_C1",
                "citation_valid": True
            }
            for field in FIELDS
        }
    }


class PaperReanalysisTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(bind=engine)
        self.Session = sessionmaker(bind=engine)
        with self.Session() as db:
            db.add(Paper(id=1, title="Test paper"))
            db.commit()

    @patch("app.routes.papers.get_paper_vector_count", return_value=5)
    @patch("app.routes.papers.analyze_paper")
    def test_reanalysis_updates_single_database_record(
        self,
        analyze,
        vector_count
    ):
        with self.Session() as db:
            analyze.return_value = analysis_result("first")
            first = analyze_research_paper(1, db)
            analyze.return_value = analysis_result("second")
            second = analyze_research_paper(1, db)

            records = db.query(PaperAnalysis).filter(
                PaperAnalysis.paper_id == 1
            ).all()
            self.assertEqual(len(records), 1)
            self.assertEqual(first["analysis_id"], second["analysis_id"])
            self.assertEqual(
                records[0].objective["value"],
                "second objective"
            )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.literature_search_agent import (
    normalize_crossref_item,
    search_literature
)
from app.database.database import Base, get_db
from app.database.models import Paper
from app.main import app as main_app
from app.routes.literature import router


class LiteratureSearchAgentTests(unittest.TestCase):
    def test_crossref_result_is_normalized(self):
        result = normalize_crossref_item({
            "DOI": "10.1000/example",
            "title": ["<i>Evidence</i> grounded research"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "published-online": {"date-parts": [[2025, 1, 2]]},
            "abstract": "<jats:p>A useful abstract.</jats:p>",
            "URL": "https://doi.org/10.1000/example",
            "container-title": ["Research Journal"],
            "type": "journal-article"
        })
        self.assertEqual(result["title"], "Evidence grounded research")
        self.assertEqual(result["authors"], ["Ada Lovelace"])
        self.assertEqual(result["year"], 2025)
        self.assertEqual(result["abstract"], "A useful abstract.")
        self.assertTrue(result["result_id"].startswith("CR_"))

    def test_search_uses_one_crossref_request_and_year_filters(self):
        def handler(request):
            self.assertEqual(request.url.params["query.bibliographic"], "AI")
            self.assertEqual(request.url.params["rows"], "3")
            self.assertEqual(
                request.url.params["filter"],
                "from-pub-date:2020,until-pub-date:2025"
            )
            return httpx.Response(200, json={
                "message": {"items": [{"title": ["Paper"]}]}
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            results = search_literature(
                "AI", limit=3, from_year=2020, to_year=2025,
                client=client
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "Crossref")


class LiteratureRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        cls.Session = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)
        test_app = FastAPI()
        test_app.include_router(router)

        def override_db():
            with cls.Session() as db:
                yield db

        test_app.dependency_overrides[get_db] = override_db
        cls.client = TestClient(test_app)

    def setUp(self):
        with self.Session() as db:
            db.query(Paper).delete()
            db.commit()

    @patch("app.routes.literature.search_literature")
    def test_search_endpoint_returns_normalized_results(self, search):
        search.return_value = [{
            "result_id": "CR_1", "title": "Paper", "authors": [],
            "year": 2025, "abstract": None, "doi": None,
            "url": None, "source": "Crossref", "venue": None,
            "publication_type": "journal-article"
        }]
        response = self.client.post(
            "/literature/search", json={"query": "evidence", "limit": 5}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)
        search.assert_called_once_with(
            query="evidence", limit=5, from_year=None, to_year=None
        )

    def test_import_endpoint_creates_and_deduplicates_paper(self):
        payload = {"papers": [{
            "result_id": "CR_1", "title": "Imported Paper",
            "authors": ["Ada Lovelace"], "year": 2025,
            "abstract": "Abstract", "doi": "10.1000/example",
            "url": "https://doi.org/10.1000/example",
            "source": "Crossref"
        }]}
        first = self.client.post("/literature/import", json=payload)
        second = self.client.post("/literature/import", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["imported_count"], 1)
        self.assertEqual(second.json()["skipped_count"], 1)
        with self.Session() as db:
            self.assertEqual(db.query(Paper).count(), 1)
            self.assertEqual(db.query(Paper).one().authors, "Ada Lovelace")

    def test_literature_routes_are_in_main_openapi(self):
        paths = main_app.openapi()["paths"]
        self.assertIn("/literature/search", paths)
        self.assertIn("post", paths["/literature/search"])
        self.assertIn("/literature/import", paths)
        self.assertIn("post", paths["/literature/import"])


if __name__ == "__main__":
    unittest.main()

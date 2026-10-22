# Evidence-Grounded-Multi-Agent-Research-System

Evidence-grounded multi-agent AI system for automated literature review, cross-paper analysis, research-gap identification, and citation-backed research synthesis.

---

## Overview

The **Evidence-Grounded Multi-Agent Research System** is an end-to-end framework designed to streamline academic and scientific research workflows. It extracts, processes, vectors, and analyzes research papers to generate verifiable, citation-backed literature reviews and highlight unexplored research directions.

### Key Capabilities

- **PDF Ingestion & Text Processing**: High-fidelity PDF text extraction, cleaning, and semantic chunking with document metadata preservation.
- **Vector Search & Grounded Retrieval (RAG)**: Persistent vector storage powered by ChromaDB and dense embeddings for semantic retrieval.
- **Cross-Paper Analysis**: Multi-agent comparative analysis comparing methodologies, datasets, benchmarks, and findings across papers.
- **Research Gap Identification**: LLM-guided detection of unanswered questions, contradictory findings, and potential opportunities for future research.
- **Automated Literature Synthesis**: Grounded research report generation referencing source papers and citations.

---

## Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic v2
- **Vector Database**: ChromaDB
- **Database / ORM**: SQLite, SQLAlchemy
- **LLM & Embeddings**: Google Gemini API (`google-genai`), Sentence Transformers
- **PDF & Text Extraction**: PyMuPDF (`pymupdf`)
- **Testing**: Pytest

---

## Project Structure

```
evidence_grounded_research_system/
├── backend/
│   ├── app/
│   │   ├── database/          # Database connection, schemas, and models
│   │   ├── routes/            # FastAPI route controllers (papers, comparison, gaps, literature)
│   │   ├── services/          # Core business logic (RAG, PDF, LLM, Chunking, Vector DB)
│   │   ├── utils/             # Utility functions and helpers
│   │   ├── config.py          # Configuration and environment settings
│   │   └── main.py            # Application entrypoint
│   ├── data/
│   │   ├── papers/            # Input research papers (PDF)
│   │   ├── extracted_text/    # Parsed and extracted text files
│   │   └── vector_db/         # Persistent vector store (ChromaDB)
│   ├── tests/                 # Automated test suite
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment variable template
├── docs/                      # Documentation and architecture diagrams
├── frontend/                  # Web application UI
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Google Gemini API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SRSPSVaishnaviKapila/Evidence-Grounded-Multi-Agent-Research-System.git
   cd Evidence-Grounded-Multi-Agent-Research-System
   ```

2. **Create and activate a virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the `backend/` directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Interactive API Documentation**:
   Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

---

## Running Tests

Run the test suite from the `backend/` directory:
```bash
pytest
```

---

## License

MIT License


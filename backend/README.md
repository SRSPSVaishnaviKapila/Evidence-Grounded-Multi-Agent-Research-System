# Backend - Evidence-Grounded Multi-Agent Research System

FastAPI backend providing REST endpoints, PDF processing, semantic embeddings, vector retrieval, and LLM-powered multi-agent research analysis.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and supply GEMINI_API_KEY
   ```

4. Start server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Testing

```bash
pytest
```

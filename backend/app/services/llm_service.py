import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

from pydantic import BaseModel


load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


MODEL_NAME = "gemini-2.5-flash"


# =========================================================
# NORMAL TEXT GENERATION
# =========================================================

def generate_answer(
    prompt: str
) -> str:

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return "No answer was generated."

    except errors.ServerError as error:

        raise RuntimeError(
            f"Gemini service unavailable: {str(error)}"
        )


# =========================================================
# STRUCTURED JSON GENERATION
# =========================================================

def generate_structured_answer(
    prompt: str,
    response_schema: type[BaseModel]
):
    """
    Generate structured JSON output using
    a Pydantic response schema.
    """

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1
            )
        )

        # Preferred method
        if response.parsed:
            return response.parsed

        # Fallback
        if response.text:
            return response_schema.model_validate_json(
                response.text
            )

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    except errors.ServerError as error:

        raise RuntimeError(
            f"Gemini service unavailable: {str(error)}"
        )
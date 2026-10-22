import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    """

    document = pymupdf.open(pdf_path)

    extracted_text = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        extracted_text.append(
            f"\n--- Page {page_number} ---\n{text}"
        )

    document.close()

    return "\n".join(extracted_text)


def save_extracted_text(
    text: str,
    output_path: str
):
    """
    Save extracted PDF text into a .txt file.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)
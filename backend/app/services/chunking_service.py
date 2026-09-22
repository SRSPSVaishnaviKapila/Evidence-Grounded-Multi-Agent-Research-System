import re


def split_text_by_pages(text: str):
    """
    Split extracted PDF text using page markers.
    """

    pattern = r"--- Page (\d+) ---"

    parts = re.split(pattern, text)

    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = parts[i + 1].strip()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


def create_chunks(
    text: str,
    paper_id: int,
    chunk_size: int = 1200,
    overlap: int = 200
):
    """
    Create overlapping chunks while preserving page numbers.
    """

    pages = split_text_by_pages(text)

    chunks = []

    chunk_counter = 1

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        start = 0

        while start < len(page_text):

            end = start + chunk_size

            chunk_text = page_text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": f"P{paper_id}_C{chunk_counter}",
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "text": chunk_text
                })

                chunk_counter += 1

            start += chunk_size - overlap

    return chunks
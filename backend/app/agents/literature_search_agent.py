import hashlib
import html
import re
from typing import Any, Optional

import httpx


CROSSREF_WORKS_URL = "https://api.crossref.org/works"
CROSSREF_USER_AGENT = (
    "EvidenceGroundedResearchSystem/1.0"
)


class LiteratureSearchError(RuntimeError):
    pass


def _first(value: Any) -> Optional[str]:
    if isinstance(value, list) and value:
        value = value[0]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _plain_text(value: Any) -> Optional[str]:
    text = _first(value)
    if not text:
        return None
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _authors(item: dict) -> list[str]:
    names = []
    for author in item.get("author") or []:
        name = " ".join(
            part for part in (
                str(author.get("given") or "").strip(),
                str(author.get("family") or "").strip()
            ) if part
        )
        if name:
            names.append(name)
    return names


def _publication_year(item: dict) -> Optional[int]:
    for field_name in ("published-print", "published-online", "published", "issued"):
        date_parts = (item.get(field_name) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def normalize_crossref_item(item: dict) -> Optional[dict]:
    title = _plain_text(item.get("title"))
    if not title:
        return None
    doi = _first(item.get("DOI"))
    url = _first(item.get("URL"))
    stable_value = doi or url or title.casefold()
    result_id = "CR_" + hashlib.sha256(
        stable_value.encode("utf-8")
    ).hexdigest()[:16]
    return {
        "result_id": result_id,
        "title": title,
        "authors": _authors(item),
        "year": _publication_year(item),
        "abstract": _plain_text(item.get("abstract")),
        "doi": doi,
        "url": url,
        "source": "Crossref",
        "venue": _plain_text(item.get("container-title")),
        "publication_type": _first(item.get("type"))
    }


def search_literature(
    query: str,
    limit: int = 10,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
    client: Optional[httpx.Client] = None
) -> list[dict]:
    """Search Crossref and return domain-independent normalized metadata."""

    params: dict[str, Any] = {
        "query.bibliographic": query.strip(),
        "rows": limit,
        "select": (
            "DOI,title,author,published-print,published-online,published,"
            "issued,abstract,URL,container-title,type"
        )
    }
    filters = []
    if from_year is not None:
        filters.append(f"from-pub-date:{from_year}")
    if to_year is not None:
        filters.append(f"until-pub-date:{to_year}")
    if filters:
        params["filter"] = ",".join(filters)

    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(15.0),
        headers={"User-Agent": CROSSREF_USER_AGENT}
    )
    try:
        response = active_client.get(CROSSREF_WORKS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except (ValueError, KeyError, TypeError) as error:
        raise LiteratureSearchError(
            "Crossref returned an invalid response."
        ) from error
    finally:
        if owns_client:
            active_client.close()

    items = (payload.get("message") or {}).get("items")
    if not isinstance(items, list):
        raise LiteratureSearchError("Crossref response did not contain works.")
    normalized = [normalize_crossref_item(item) for item in items]
    return [item for item in normalized if item is not None][:limit]

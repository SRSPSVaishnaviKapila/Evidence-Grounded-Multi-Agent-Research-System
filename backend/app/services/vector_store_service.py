import chromadb

from app.services.embedding_service import (
    generate_embeddings,
    generate_query_embedding
)


CHROMA_PATH = "data/vector_db"
COLLECTION_NAME = "research_paper_chunks"


# =========================================================
# CHROMA DATABASE
# =========================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# STORE CHUNKS
# =========================================================

def store_chunks(chunks: list[dict]):
    """
    Generate embeddings and store chunks
    inside ChromaDB.
    """

    if not chunks:
        return 0

    texts = []
    ids = []
    metadatas = []

    for chunk in chunks:

        texts.append(
            chunk["text"]
        )

        ids.append(
            str(chunk["chunk_id"])
        )

        metadatas.append({
            "paper_id": int(
                chunk["paper_id"]
            ),
            "page_number": int(
                chunk["page_number"]
            ),
            "chunk_id": str(
                chunk["chunk_id"]
            )
        })

    # Generate embeddings
    embeddings = generate_embeddings(
        texts
    )

    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            "No embeddings were generated."
        )

    # Store/update records in ChromaDB
    collection.upsert(
        ids=ids,

        embeddings=[
            [float(value) for value in embedding]
            for embedding in embeddings
        ],

        documents=texts,

        metadatas=metadatas
    )

    return len(ids)


# =========================================================
# SEARCH CHUNKS
# =========================================================

def search_chunks(
    query: str,
    paper_id: int | None = None,
    n_results: int = 5
):
    """
    Search for chunks semantically related
    to the given query.
    """

    query_embedding = (
        generate_query_embedding(
            query
        )
    )

    query_arguments = {

        "query_embeddings": [
            [float(value) for value in query_embedding]
        ],

        "n_results":
            n_results,

        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    # Search only within one paper
    if paper_id is not None:

        query_arguments["where"] = {
            "paper_id": int(paper_id)
        }

    results = collection.query(
        **query_arguments
    )

    # -----------------------------------------
    # Read results
    # -----------------------------------------

    ids = results.get(
        "ids",
        []
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    distances = results.get(
        "distances",
        []
    )

    # No results found
    if (
        not documents
        or not documents[0]
    ):
        return []

    formatted_results = []

    # -----------------------------------------
    # Format results
    # -----------------------------------------

    for i in range(
        len(documents[0])
    ):

        formatted_results.append({

            "chunk_id":
                ids[0][i],

            "paper_id":
                metadatas[0][i][
                    "paper_id"
                ],

            "page_number":
                metadatas[0][i][
                    "page_number"
                ],

            "text":
                documents[0][i],

            "distance":
                distances[0][i]
        })

    return formatted_results


# =========================================================
# TOTAL VECTOR COUNT
# =========================================================

def get_collection_count():
    """
    Return total number of chunks
    currently stored in ChromaDB.
    """

    return collection.count()


# =========================================================
# PAPER-SPECIFIC VECTOR COUNT
# =========================================================

def get_paper_vector_count(
    paper_id: int
):
    """
    Return number of vectors belonging
    to a specific paper.
    """

    results = collection.get(
        where={
            "paper_id": int(paper_id)
        }
    )

    return len(
        results.get(
            "ids",
            []
        )
    )


# =========================================================
# DELETE PAPER VECTORS
# =========================================================

def delete_paper_vectors(
    paper_id: int
):
    """
    Delete all vectors belonging
    to a particular paper.
    """

    collection.delete(
        where={
            "paper_id": int(paper_id)
        }
    )

    return True

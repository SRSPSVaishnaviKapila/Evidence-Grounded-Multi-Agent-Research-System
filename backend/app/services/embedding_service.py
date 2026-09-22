from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction
)


# Chroma's local default embedding model:
# all-MiniLM-L6-v2
embedding_model = DefaultEmbeddingFunction()


def generate_embeddings(
    texts: list[str]
):
    """
    Generate embeddings for multiple chunks
    without using PyTorch.
    """

    if not texts:
        return []

    embeddings = embedding_model(
        texts
    )

    return embeddings


def generate_query_embedding(
    query: str
):
    """
    Generate embedding for one query.
    """

    embeddings = embedding_model(
        [query]
    )

    return embeddings[0]
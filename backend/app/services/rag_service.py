from app.services.vector_store_service import (
    search_chunks
)

from app.services.llm_service import (
    generate_answer
)


def build_context(
    retrieved_chunks: list[dict]
) -> str:
    """
    Convert retrieved chunks into
    evidence context for the LLM.
    """

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_part = f"""
Evidence {index}
Paper ID: {chunk["paper_id"]}
Page: {chunk["page_number"]}
Chunk ID: {chunk["chunk_id"]}

Text:
{chunk["text"]}
"""

        context_parts.append(
            context_part.strip()
        )

    return "\n\n".join(
        context_parts
    )


def answer_question(
    paper_id: int,
    question: str,
    top_k: int = 5
):
    """
    Complete RAG pipeline:

    question
        -> retrieval
        -> evidence context
        -> LLM
        -> grounded answer
    """

    # Retrieve relevant evidence
    retrieved_chunks = search_chunks(
        query=question,
        paper_id=paper_id,
        n_results=top_k
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "No relevant evidence "
                "was found in this paper."
            ),
            "evidence": []
        }

    # Build context
    context = build_context(
        retrieved_chunks
    )

    # Evidence-grounded prompt
    prompt = f"""
You are an evidence-grounded research assistant.

Your task is to answer a question about a
research paper using ONLY the evidence
provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the evidence is insufficient, say:
   "The provided evidence is insufficient
   to answer this question."
4. Give a clear and concise academic answer.
5. Cite the supporting evidence using the
   exact chunk ID and page number.
6. Do not cite evidence that does not
   support the answer.

Question:
{question}

Retrieved Evidence:
{context}

Return an answer in this format:

Answer:
<answer>

Supporting Evidence:
- <chunk_id>, Page <page_number>
"""

    # Ask LLM
    answer = generate_answer(
        prompt
    )

    # Keep evidence separate as well
    evidence = []

    for chunk in retrieved_chunks:

        evidence.append({
            "chunk_id":
                chunk["chunk_id"],

            "paper_id":
                chunk["paper_id"],

            "page_number":
                chunk["page_number"],

            "text":
                chunk["text"],

            "distance":
                chunk["distance"]
        })

    return {
        "answer": answer,
        "evidence": evidence
    }
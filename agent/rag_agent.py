"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    """Format retrieved docs with citations.

    `page_content` holds `chunk_to_embed` (the embedding source column) —
    display `chunk_to_retrieve` from metadata instead, since that's the
    clean text meant to be shown to the LLM/user, not the embedding input.
    """
    if not docs:
        return ""
    lines = []
    for doc in docs:
        text = doc.metadata.get("chunk_to_retrieve", doc.page_content)
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        if isinstance(page, float) and page.is_integer():
            page = int(page)
        lines.append(f"[source: {source}, p.{page}] {text}")
    return "\n\n".join(lines)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        plan = state["plan"]
        index = state["current_step_index"]
        step = plan[index]

        docs = retriever.invoke(step)
        context = format_docs(docs)

        if not context:
            fact = "not found in documents"
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(content=f"Step: {step}\n\nRetrieved context:\n{context}"),
                ]
            )
            fact = response.content.strip()

        step_results = state.get("step_results", []) + [fact]
        return {"step_results": step_results, "current_step_index": index + 1}

    return rag_agent

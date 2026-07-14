"""Vector Search retriever factory (Task 1.4 support / rag/store.py)."""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_embed"
CITATION_COLUMNS = ["chunk_id", "chunk_to_retrieve", "source", "page"]


def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()
    if not s["vs_endpoint"] or not s["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set "
            "(in .env locally or the endpoint's environment_vars when deployed)."
        )
    return DatabricksVectorSearch(
        index_name=s["vs_index"],
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})

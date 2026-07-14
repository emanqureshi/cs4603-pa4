"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirrors PA2 Part 1: parse -> chunk -> Delta table -> Delta Sync Vector Search index.

Typical notebook usage:

    from rag.ingest import ingest

    ingest(spark, volume_path="/Volumes/main/default/pa4/annual_report.pdf")

`ingest()` is a thin convenience wrapper around `build_chunks_table` +
`create_index` + `wait_for_index_ready`; call the pieces individually if you
want to re-run chunking without re-creating the index, or vice versa.
"""

from __future__ import annotations

import time

from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse `volume_path` (a PDF in a UC volume) and chunk it into `chunks_table`.

    `ai_parse_document` extracts per-page structured text from the PDF;
    `ai_prep_search` splits that text into retrieval-sized chunks. The
    resulting Delta table has exactly the columns the Vector Search Delta Sync
    index and `rag/store.py::get_vector_store()` expect: `chunk_id` (primary
    key), `chunk_to_embed` (what gets embedded — used as the embedding source
    column), `chunk_to_retrieve` (the text actually shown to the retriever /
    LLM / citations), plus `source`/`page` for citations.
    """
    source_name = volume_path.rsplit("/", 1)[-1]

    spark.sql(f"""
        CREATE OR REPLACE TABLE {chunks_table} AS
        WITH parsed AS (
            SELECT ai_parse_document(content) AS parsed_doc
            FROM read_files('{volume_path}', format => 'binaryFile')
        ),
        chunked AS (
            SELECT explode(ai_prep_search(parsed_doc)) AS chunk
            FROM parsed
        )
        SELECT
            uuid() AS chunk_id,
            chunk.chunk_text AS chunk_to_retrieve,
            chunk.chunk_text AS chunk_to_embed,
            '{source_name}' AS source,
            chunk.page_number AS page
        FROM chunked
    """)

    # Delta Sync indexes read new rows via Change Data Feed.
    spark.sql(f"""
        ALTER TABLE {chunks_table}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)


def create_index(chunks_table: str) -> None:
    """Create a STANDARD Vector Search endpoint and a TRIGGERED Delta Sync
    index over `chunks_table`, using the endpoint/index names and embedding
    model from `.env` (`config.get_settings()`).
    """
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    endpoint_name = s["vs_endpoint"]
    index_name = s["vs_index"]
    if not endpoint_name or not index_name:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set in .env "
            "before creating the index."
        )

    client = VectorSearchClient()

    try:
        client.get_endpoint(endpoint_name)
    except Exception:
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    _wait_for_endpoint_online(client, endpoint_name)

    try:
        client.get_index(endpoint_name, index_name)
        index_exists = True
    except Exception:
        index_exists = False

    if not index_exists:
        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=chunks_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_embed",
            embedding_model_endpoint_name=s["embeddings"],
        )
    wait_for_index_ready(index_name, endpoint_name=endpoint_name)


def _wait_for_endpoint_online(client, endpoint_name: str, timeout_s: int = 600, poll_s: int = 15) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = client.get_endpoint(endpoint_name).get("endpoint_status", {}).get("state")
        if state == "ONLINE":
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Vector Search endpoint '{endpoint_name}' did not come ONLINE within {timeout_s}s")


def wait_for_index_ready(
    index_name: str, endpoint_name: str | None = None, timeout_s: int = 1800, poll_s: int = 30
) -> None:
    """Block until the Delta Sync index finishes its initial sync and reports READY."""
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    endpoint_name = endpoint_name or s["vs_endpoint"]
    client = VectorSearchClient()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = client.get_index(endpoint_name, index_name).describe().get("status", {})
        if status.get("ready"):
            return
        if status.get("detailed_state") == "FAILED":
            raise RuntimeError(f"Index '{index_name}' sync failed: {status}")
        time.sleep(poll_s)
    raise TimeoutError(f"Index '{index_name}' did not become READY within {timeout_s}s")


def verify_index(query: str = "net revenue", k: int = 3) -> list[dict]:
    """Run a similarity-search smoke test against the index (Task 0.3, step 4)."""
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    client = VectorSearchClient()
    index = client.get_index(s["vs_endpoint"], s["vs_index"])
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_id", "chunk_to_retrieve", "source", "page"],
        num_results=k,
    )
    return results.get("result", {}).get("data_array", [])


def ingest(spark, volume_path: str, chunks_table: str | None = None) -> None:
    """Convenience wrapper: parse+chunk the PDF, then create/sync the index."""
    s = get_settings()
    chunks_table = chunks_table or s["source_table"]
    if not chunks_table:
        raise OSError("Pass chunks_table explicitly or set SOURCE_TABLE in .env.")
    build_chunks_table(spark, volume_path, chunks_table)
    create_index(chunks_table)

"""Shared configuration and client factories for the Document Analyst.

Centralising credential loading here keeps every other module free of
`os.environ` calls and makes the LLM / retriever easy to mock in tests.
All clients are created lazily so importing the package never requires
network access or credentials — only *invoking* a node does.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise OSError(
            f"Missing required environment variable: {name}. "
            "Set it in your .env (local) or the endpoint secret scope (deployed)."
        )
    return val


def get_settings() -> dict[str, str]:
    """Return all configuration values, validating the required ones."""
    return {
        "host": _require("DATABRICKS_HOST").rstrip("/"),
        "token": _require("DATABRICKS_TOKEN"),
        "model": _require("DATABRICKS_MODEL"),
        "embeddings": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
        "vs_endpoint": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
        "vs_index": os.environ.get("VECTOR_SEARCH_INDEX", ""),
        "source_table": os.environ.get("SOURCE_TABLE", ""),
        "uc_catalog": os.environ.get("UC_CATALOG", "main"),
        "uc_schema": os.environ.get("UC_SCHEMA", "default"),
        "serving_endpoint_name": os.environ.get("SERVING_ENDPOINT_NAME", "document-analyst"),
        "secret_scope": os.environ.get("SECRET_SCOPE", "cs4603-deploy"),
    }


@lru_cache(maxsize=1)
def get_chat_llm(temperature: float = 0.0):
    """Configured ChatOpenAI client pointed at Databricks Model Serving.

    We use the OpenAI-compatible surface (same as PA1–PA3) so the deployed
    endpoint speaks the same protocol the client SDK expects.
    """
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(
        model=s["model"],
        api_key=s["token"],
        base_url=f"{s['host']}/serving-endpoints",
        temperature=temperature,
    )

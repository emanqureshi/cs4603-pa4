"""MLflow models-from-code definition (Task 2.1).

Self-contained so MLflow can serialise it: imports the Part 1 graph plus
production clients (LLM, Vector Search retriever, MCP tools) and ends with
`mlflow.models.set_model(graph)`. `code_paths` in deployment/deploy.py ships
the `agent`/`rag`/`tools` packages and `config.py` alongside this file so the
serving container can import them.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import mlflow

from agent.graph import build_graph
from config import get_chat_llm, get_settings
from rag.store import get_retriever

# Validate required env vars (DATABRICKS_HOST/TOKEN/MODEL) at import time so a
# misconfigured serving container fails here with a clear message, instead of
# a generic DEPLOYMENT_FAILED with no cause in the logs.
get_settings()

# tools=None lets build_graph() resolve the MCP server path itself via
# agent.graph.load_mcp_tools()'s import-based lookup, which stays correct
# regardless of how MLflow's packaging sandbox lays out code_paths.
graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
)

mlflow.models.set_model(graph)

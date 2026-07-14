"""Bonus B model definition — wraps the graph for Agent Framework compatibility.

`databricks.agents.deploy()` validates the served model's *output* schema and
requires it to be `ChatCompletionResponse` or `StringResponse`. Part 2's
`agent_model.py` serves the full `AnalystState` dict (`messages`, `plan`,
`step_results`, `current_step_index`, `next_agent`, `final_answer`) — exactly
what the manual serving path in Task 2.1/2.2 is designed to tolerate (it just
reads `messages[-1]`), but Agent Framework rejects it outright with:

    ValueError: The model's schema is not compatible with Agent Framework.
    The output schema must be either ChatCompletionResponse or StringResponse.

This file reuses the exact same `build_graph()` (with every Task 2.3 fix —
lazy MCP loading, the stdio `errlog` patch, sys.executable subprocess
resolution — already baked in) but wraps it in a `RunnableLambda` that
returns only the final answer as a plain string, which MLflow infers a
`StringResponse`-compatible schema from.

Must import cleanly:  python -c "import deployment.agent_model_chat"
"""

from __future__ import annotations

import mlflow
from langchain_core.runnables import RunnableLambda

from agent.graph import build_graph
from config import get_chat_llm, get_settings
from rag.store import get_retriever

# Validate required env vars at import time — same reasoning as agent_model.py.
get_settings()

_graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
)


def _invoke(input_data) -> str:
    """Accept either shape MLflow might call this with: the full
    `{"messages": [...]}` dict our graph expects, or — what MLflow's
    chat-model input adaptation actually passes, inferred from the
    `input_example`'s `messages` schema — a bare list of messages.
    """
    messages = input_data.get("messages", []) if isinstance(input_data, dict) else input_data
    result = _graph.invoke({"messages": messages})
    return result["messages"][-1].content


chat_model = RunnableLambda(_invoke)

mlflow.models.set_model(chat_model)

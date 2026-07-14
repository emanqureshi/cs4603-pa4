"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

Reuses Part 2's exact logging pipeline (code_paths, pinned pip requirements,
Python-3.11 conda_env — everything Task 2.3 needed to get a real container to
build) via `deploy.py::log_and_register()`, but points it at
`agent_model_chat.py` instead of `agent_model.py`: `agents.deploy()`'s Agent
Framework validates the served model's output schema and requires
`ChatCompletionResponse`/`StringResponse`, which our full `AnalystState`
output (Part 2's contract) doesn't satisfy — `agent_model_chat.py` wraps the
same graph to return just the final answer string instead.

Must run under Python 3.11 (see deploy.py's `_CONDA_ENV` comment for why):
    uv run --python 3.11 python deployment/deploy_agents.py
"""

from __future__ import annotations

from deployment.deploy import log_and_register

CHAT_MODEL_NAME = "document_analyst_chat"


def main() -> None:
    from databricks import agents

    uc_name, version = log_and_register(
        model_name=CHAT_MODEL_NAME,
        lc_model_filename="agent_model_chat.py",
        run_name="document-analyst-chat-deploy",
    )

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
    )
    print("Endpoint name:", deployment.endpoint_name)
    print("Review app URL:", deployment.review_app_url)


if __name__ == "__main__":
    main()

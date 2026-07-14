"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py
"""

from __future__ import annotations

import os

import mlflow

from config import get_settings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "document_analyst"

_PIP_REQUIREMENTS = [
    "langgraph==1.2.9",
    "langchain==1.3.13",
    "langchain-core==1.4.9",
    "langchain-openai==1.3.5",
    "databricks-langchain==0.20.0",
    "databricks-vectorsearch==0.75",
    "langchain-mcp-adapters==0.3.0",
    "mcp==1.28.1",
    "mlflow==3.14.0",
    "openai==2.45.0",
    "python-dotenv==1.2.2",
]

# Explicitly pin the container's Python version to 3.11 rather than letting
# MLflow infer it from whatever interpreter happens to run this script.
# Confirmed by inspecting our own logged models' python_env.yaml: versions
# logged from Python 3.14 and 3.13.7 both produced serving containers that
# never progressed past "Container creation pending" (no error, no build
# logs, indefinitely) — Databricks Serving's build environment appears not
# to reliably support Python newer than ~3.11 yet.
_CONDA_ENV = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": ["python=3.11", "pip", {"pip": _PIP_REQUIREMENTS}],
}


def log_and_register(
    model_name: str = MODEL_NAME,
    lc_model_filename: str = "agent_model.py",
    run_name: str = "document-analyst-deploy",
) -> tuple[str, str]:
    """Log + register the model. Defaults reproduce Task 2.2 exactly;
    `model_name`/`lc_model_filename` are overridable so Bonus B
    (deploy_agents.py) can reuse this same logging pipeline (code_paths,
    pip pins, Python 3.11 conda_env) to log its Agent-Framework-compatible
    wrapper (`agent_model_chat.py`) under a separate UC model name, without
    duplicating any of it or touching Part 2's own behavior.
    """
    settings = get_settings()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    from databricks.sdk import WorkspaceClient

    user_name = WorkspaceClient(host=settings["host"], token=settings["token"]).current_user.me().user_name
    mlflow.set_experiment(f"/Users/{user_name}/{model_name}")

    uc_name = f"{settings['uc_catalog']}.{settings['uc_schema']}.{model_name}"

    with mlflow.start_run(run_name=run_name):
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(_REPO_ROOT, "deployment", lc_model_filename),
            name="agent",
            code_paths=[
                os.path.join(_REPO_ROOT, "agent"),
                os.path.join(_REPO_ROOT, "rag"),
                os.path.join(_REPO_ROOT, "tools"),
                os.path.join(_REPO_ROOT, "config.py"),
            ],
            conda_env=_CONDA_ENV,
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )
    print(f"Logged model: {model_info.model_uri}")

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    settings = get_settings()
    endpoint_name = settings["serving_endpoint_name"]
    scope = settings["secret_scope"]

    w = WorkspaceClient(host=settings["host"], token=settings["token"])

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=str(version),
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "DATABRICKS_HOST": f"{{{{secrets/{scope}/DATABRICKS_HOST}}}}",
            "DATABRICKS_TOKEN": f"{{{{secrets/{scope}/DATABRICKS_TOKEN}}}}",
            "DATABRICKS_MODEL": f"{{{{secrets/{scope}/DATABRICKS_MODEL}}}}",
            # Not secrets — the retriever needs these to reach the Vector
            # Search index; plaintext is fine.
            "EMBEDDINGS_ENDPOINT": settings["embeddings"],
            "VECTOR_SEARCH_ENDPOINT": settings["vs_endpoint"],
            "VECTOR_SEARCH_INDEX": settings["vs_index"],
        },
    )

    try:
        w.serving_endpoints.get(endpoint_name)
        exists = True
    except NotFound:
        exists = False

    if exists:
        print(f"Updating existing endpoint '{endpoint_name}' to version {version}...")
        w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=[served_entity])
    else:
        print(f"Creating endpoint '{endpoint_name}'...")
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[served_entity]),
        )

    url = f"{settings['host']}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint READY: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)

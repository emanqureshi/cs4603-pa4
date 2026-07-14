"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

Reuses the exact same models-from-code definition + code_paths as Part 2
(deployment/deploy.py::log_and_register) and only swaps the final
WorkspaceClient endpoint-creation step for a single agents.deploy() call,
which also provisions a Review App.

Run:  uv run python deployment/deploy_agents.py
"""

from __future__ import annotations

from deployment.deploy import log_and_register


def main() -> None:
    from databricks import agents

    uc_name, version = log_and_register()

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
    )
    print("Endpoint name:", deployment.endpoint_name)
    print("Review app URL:", deployment.review_app_url)


if __name__ == "__main__":
    main()

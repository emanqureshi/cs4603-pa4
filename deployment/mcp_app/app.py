"""Bonus C — standalone MCP server as a Databricks App.

Reuses the GIVEN tool definitions from tools/mcp_server.py unmodified, but
serves them over HTTP (streamable-http transport) instead of stdio, so the
tool server can be deployed, scaled, and monitored independently of the
model (see README.md's Bonus C section for why this decouples the tool
server from the model container).
"""

from __future__ import annotations

import os
import sys

# Databricks Apps runs this as `python deployment/mcp_app/app.py`, which
# makes Python set sys.path[0] to this script's own directory
# (deployment/mcp_app/), not the repo root — so `tools` (a sibling of
# deployment/) isn't importable by default. This worked locally only
# because `uv sync` installs the project's own packages (including `tools`)
# into the venv as editable packages; Databricks Apps' plain
# requirements.txt install does not do that. Insert the repo root
# explicitly so the import below resolves regardless of how the process
# was launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.mcp_server import mcp

if __name__ == "__main__":
    # Databricks Apps provides the port via $DATABRICKS_APP_PORT and expects
    # the app to bind all interfaces, not just localhost (FastMCP's default).
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http")

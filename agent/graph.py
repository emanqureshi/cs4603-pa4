"""Full Document Analyst graph (Tasks 1.5 + 1.7)."""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _default_server_path() -> str:
    """Resolve tools/mcp_server.py's real path via the import system.

    A `__file__`-relative guess (assuming `tools/` sits next to `agent/`)
    breaks once MLflow's models-from-code packaging copies `code_paths` into
    a sandbox that doesn't preserve the source repo's directory nesting.
    Resolving through `import tools.mcp_server` is correct wherever the
    `tools` package actually ends up, since that's the same mechanism
    `from agent.graph import ...` etc. already rely on to work post-packaging.
    """
    import tools.mcp_server as mcp_server_module

    return mcp_server_module.__file__


def _run_async(coro):
    """Run `coro` to completion whether or not an event loop is already running.

    Plain `asyncio.run()` fails with "cannot be called from a running event
    loop" inside Jupyter/Databricks notebook kernels and potentially the
    MLflow serving container (both already run their own loop). Detect that
    case and run the coroutine on a fresh loop in a separate thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _has_fileno(stream) -> bool:
    try:
        stream.fileno()
        return True
    except (AttributeError, OSError, ValueError):
        return False


def _patch_mcp_stdio_errlog() -> None:
    """Force mcp.client.stdio.stdio_client's `errlog` default to a real,
    fileno-backed stream.

    `stdio_client(server, errlog: TextIO = sys.stderr)` binds its default
    once, at import time, to whatever `sys.stderr` is at that instant.
    Databricks Model Serving replaces `sys.stderr` with a logging wrapper
    (`StreamToLogger`) that has no `.fileno()`, which the MCP subprocess
    spawn needs. `stdio_client` is decorated with `@asynccontextmanager`,
    so `stdio_client.__defaults__` is the *wrapper's* defaults (empty) —
    the real default lives on `stdio_client.__wrapped__` (the attribute
    `functools.wraps` sets to the original function), which is the actual
    object patching needs to target.
    """
    import os

    import mcp.client.stdio as stdio_module

    target = getattr(stdio_module.stdio_client, "__wrapped__", stdio_module.stdio_client)
    if target.__defaults__ and not _has_fileno(target.__defaults__[0]):
        fallback = sys.__stderr__ if sys.__stderr__ and _has_fileno(sys.__stderr__) else open(os.devnull, "w")
        target.__defaults__ = (fallback,)


def _get_databricks_app_token(host: str, sp_client_id: str, sp_client_secret: str) -> str:
    """Mint a short-lived OAuth access token for calling a Databricks App.

    Databricks Apps' auth proxy doesn't accept plain personal access tokens —
    confirmed empirically: every request got redirected to the OAuth login
    page (a 302 to /oidc/oauth2/v2.0/authorize) regardless of the
    Authorization header sent. A service principal's OAuth2
    client-credentials grant is the actual supported programmatic path,
    using the `all-apis` scope.
    """
    import httpx

    resp = httpx.post(
        f"{host}/oidc/v1/token",
        auth=(sp_client_id, sp_client_secret),
        data={"grant_type": "client_credentials", "scope": "all-apis"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def _load_mcp_tools_async(server_path: str | None, server_url: str | None, bearer_token: str | None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if server_url:
        # Bonus C — remote HTTP MCP server (deployment/mcp_app/app.py, a
        # separate Databricks App). Decouples the tool server from the model
        # container entirely: no subprocess, no stdio, no errlog/fileno
        # fragility at all — see README.md's Bonus C section for why this is
        # the production-style alternative to Task 1.5's bundled subprocess.
        connection = {
            "analyst": {
                "url": f"{server_url.rstrip('/')}/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {},
            }
        }
    else:
        _patch_mcp_stdio_errlog()
        connection = {
            "analyst": {
                # sys.executable (not the bare string "python") so the
                # subprocess uses the exact interpreter already running,
                # regardless of PATH/pyenv/venv-activation quirks in whatever
                # shell spawned this process.
                "command": sys.executable,
                "args": [server_path],
                "transport": "stdio",
            }
        }

    client = MultiServerMCPClient(connection)
    return await client.get_tools()


def load_mcp_tools(server_path: str | None = None):
    """Connect to the MCP server and return its LangChain tools.

    Uses the remote HTTP server at $MCP_SERVER_URL (Bonus C) when configured
    — authenticated with a service-principal OAuth token (see
    `_get_databricks_app_token`), not $DATABRICKS_TOKEN — falling back to
    spawning the GIVEN stdio server (tools/mcp_server.py, Task 1.5) when it
    isn't set.
    """
    from config import get_settings

    settings = get_settings()
    server_url = settings["mcp_server_url"]
    bearer_token = None
    if server_url:
        bearer_token = _get_databricks_app_token(
            settings["host"], settings["mcp_sp_client_id"], settings["mcp_sp_client_secret"]
        )
    else:
        server_path = server_path or _default_server_path()
    return _run_async(_load_mcp_tools_async(server_path, server_url, bearer_token))


def _flatten_tool_result(result) -> str:
    """Reduce an MCP tool result to plain text.

    langchain-mcp-adapters returns MCP content blocks
    (`[{"type": "text", "text": "..."}]`) rather than a plain string;
    join any text blocks so downstream nodes get readable output.
    """
    if isinstance(result, list):
        texts = [block.get("text", "") for block in result if isinstance(block, dict) and "text" in block]
        if texts:
            return "\n".join(texts)
    return str(result)


def make_mcp_node(tools, llm, server_path: str | None = None):
    """Build the MCP tool-calling node.

    `tools=None` defers `load_mcp_tools()` (which spawns the MCP stdio
    subprocess) to the node's first actual invocation instead of graph-build
    time. Spawning a subprocess during MLflow's synchronous model-load phase
    conflicts with Databricks Serving's stdout/stderr redirection there in a
    way that survived two targeted fixes (see `_patch_mcp_stdio_errlog`);
    deferring it to request time avoids that constrained phase entirely.
    """
    _cache: dict = {"tools": tools, "llm_with_tools": None, "tools_by_name": None}

    def _ensure_loaded() -> None:
        if _cache["tools"] is None:
            _cache["tools"] = load_mcp_tools(server_path)
        if _cache["llm_with_tools"] is None:
            _cache["llm_with_tools"] = llm.bind_tools(_cache["tools"])
            _cache["tools_by_name"] = {tool.name: tool for tool in _cache["tools"]}

    def mcp_tools(state: AnalystState) -> dict:
        _ensure_loaded()
        llm_with_tools = _cache["llm_with_tools"]
        tools_by_name = _cache["tools_by_name"]

        plan = state["plan"]
        index = state["current_step_index"]
        step = plan[index]
        step_results = state.get("step_results", [])

        prior_results = "\n".join(step_results) or "(none yet)"
        prompt = f"{MCP_STEP_PROMPT}\n\nPrior results:\n{prior_results}\n\nStep: {step}"
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            result = response.content.strip() or "Error: no tool was called"
        else:
            call = tool_calls[0]
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Error: unknown tool '{call['name']}'"
            else:
                # langchain-mcp-adapters tools only implement async execution
                # (they proxy the call over the stdio MCP session), so bridge
                # to sync here rather than exposing an async node.
                result = _flatten_tool_result(_run_async(tool.ainvoke(call["args"])))

        return {
            "step_results": step_results + [str(result)],
            "current_step_index": index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()

    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()

    # tools=None is passed through to make_mcp_node, which lazily loads MCP
    # tools on the node's first invocation rather than here at build time.
    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("rag_agent", make_rag_agent(retriever, llm))
    builder.add_node("mcp_tools", make_mcp_node(tools, llm))
    builder.add_node("synthesizer", make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()

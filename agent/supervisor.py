"""Supervisor node + routing edge (Task 1.3)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        if index >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[index]
        response = llm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=step)]
        )
        decision = response.content.strip().lower()
        next_agent = MCP if MCP in decision else RAG
        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]

"""Offline smoke test for the Document Analyst graph (Bonus A test target).

Builds the graph with fake LLM / retriever / tool objects (no Databricks, no
network) and drives it through a combined retrieval + calculation query.

Run:  uv run pytest -q
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage

from agent.prompts import (
    PLANNER_PROMPT,
    RAG_EXTRACT_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIZER_PROMPT,
)


class FakeDoc:
    def __init__(self, content: str, source: str, page: int) -> None:
        self.page_content = content
        self.metadata = {"source": source, "page": page}


class FakeRetriever:
    def invoke(self, query: str):
        return [FakeDoc("Net revenue in FY2023 was 16.91 trillion.", "annual_report.pdf", 4)]


class FakeToolBoundLLM:
    """Stands in for `llm.bind_tools(tools)` — always calls growth_rate once."""

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "growth_rate",
                    "args": {"start_value": 16.91, "rate": 0.08, "years": 3},
                    "id": "call_1",
                }
            ],
        )


class FakeLLM:
    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content

        if system == PLANNER_PROMPT:
            steps = [
                "Find Meridian's net revenue for fiscal year 2023",
                "Calculate the value after 8% growth for 3 years",
            ]
            return AIMessage(content=json.dumps(steps))

        if system == SUPERVISOR_PROMPT:
            if "calculate" in human.lower() or "growth" in human.lower():
                return AIMessage(content="mcp_tools")
            return AIMessage(content="rag_agent")

        if system == RAG_EXTRACT_PROMPT:
            return AIMessage(
                content=(
                    "Meridian's net revenue in FY2023 was 16.91 trillion "
                    "[source: annual_report.pdf, p.4]"
                )
            )

        if system == SYNTHESIZER_PROMPT:
            return AIMessage(content="Combined answer citing both steps.")

        return AIMessage(content="")

    def bind_tools(self, tools):
        return FakeToolBoundLLM()


class FakeTool:
    """Mirrors langchain-mcp-adapters tools, which only support async execution."""

    name = "growth_rate"

    async def ainvoke(self, args):
        text = f"{args['start_value']} at {args['rate'] * 100:g}% CAGR for {args['years']} years = 21.30 (fake)"
        return [{"type": "text", "text": text}]


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_runs_end_to_end_offline():
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What was the revenue in 2023, and what would it be after "
                        "8% growth for 3 years?"
                    ),
                }
            ]
        }
    )

    assert result["plan"]
    assert len(result["step_results"]) == len(result["plan"])
    assert result["final_answer"]
    assert result["messages"][-1].content == result["final_answer"]

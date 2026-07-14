"""Planner node (Task 1.2)."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _parse_plan(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in planner output")
    steps = json.loads(match.group(0))
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) for s in steps):
        raise ValueError("planner output is not a non-empty list of strings")
    return steps


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        question = state["messages"][-1].content
        response = llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=question)]
        )
        try:
            plan = _parse_plan(response.content)
        except (ValueError, json.JSONDecodeError):
            plan = [question]
        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner

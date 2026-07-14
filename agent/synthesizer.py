"""Synthesizer node (Task 1.6)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        step_results = state.get("step_results", [])

        context = "\n".join(
            f"Step {i + 1}: {step}\nResult: {result}"
            for i, (step, result) in enumerate(zip(plan, step_results, strict=False))
        )

        response = llm.invoke(
            [SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=context)]
        )
        final_answer = response.content.strip()

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return synthesizer

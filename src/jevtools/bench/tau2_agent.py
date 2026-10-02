"""Full τ²-bench tasks with jevtools deciding every tool call (needs a τ²-bench install: ``pip install -e tau2-bench``).

The next-call reduction (:mod:`jevtools.bench.tau2`) scores single decisions; τ²'s own metric is whether a whole
simulated conversation ends with the right database state. :func:`agent_class` builds a τ² ``HalfDuplexAgent`` in
which jevtools decides each turn and an LLM only writes text:

- ``execute`` → the tool call;
- ``confirm`` / ``clarify`` → jevtools' own prompt (the confirm card or the question) as the assistant's message; the
  customer's next message resumes it (:meth:`jevtools.router.Router.resume`, a free-text reply);
- anything else (no tool, done, abstain, refuse) → a text reply the LLM writes from the conversation, with the
  domain policy and no tools.

Compared with τ²'s ``llm_agent`` on the same LLM and the same simulated customer, the difference in reward is what
jevtools' tool decisions change. ``python -m jevtools.bench.tau2_agent --help`` runs both.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from jevtools.bench.tau2 import NOW, READ_PREFIXES
from jevtools.context import Context, Observation
from jevtools.fallback import OpenAICompatibleTextLLM
from jevtools.policy import Outcome
from jevtools.router import Router

REPLY_SYSTEM = (
    "You are a customer service agent. Follow the policy below. In this turn you cannot call any tool: reply to the "
    "customer in plain text, using only facts from the conversation and the tool results shown in it. Never claim "
    "that an action was done unless its result is shown.\n\n# Policy\n{policy}"
)

__all__ = ["REPLY_SYSTEM", "agent_class", "main"]


def _content(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def agent_class() -> Any:
    """The ``JevtoolsAgent`` class (imports τ² on first use)."""
    from tau2.agent.base_agent import HalfDuplexAgent  # type: ignore[import-not-found]
    from tau2.data_model.message import (  # type: ignore[import-not-found]
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    class JevtoolsAgent(HalfDuplexAgent):  # type: ignore[misc]
        """jevtools decides; an LLM writes the text replies jevtools has none for."""

        def __init__(self, tools: Sequence[Any], domain_policy: str, *, backend: Any, llm: str,
                     trust: str = "reads", escalator: Any = None) -> None:  # fmt: skip
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.openai_tools = [deepcopy(t.openai_schema) for t in tools]
            names = [t["function"]["name"] for t in self.openai_tools]
            self.trusted = tuple(n for n in names if n.startswith(READ_PREFIXES)) if trust == "reads" else ()
            self.router = Router(deepcopy(self.openai_tools), backend=backend, escalator=escalator)
            self.writer = OpenAICompatibleTextLLM(llm, system=REPLY_SYSTEM.format(policy=domain_policy),
                                                  temperature=0.0, extra_body={"usage": {"include": True}})  # fmt: skip
            self.jev_cost = 0.0
            self.decisions: list[dict[str, Any]] = []

        def get_init_state(self, message_history: Sequence[Any] | None = None) -> dict[str, Any]:
            state: dict[str, Any] = {"turns": [], "obs": [], "calls": {}, "pending": None, "n": 0, "last": None}
            for message in message_history or []:
                self._ingest(message, state)
            return state

        def _ingest(self, message: Any, state: dict[str, Any]) -> None:
            messages = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            for m in messages:
                if isinstance(m, ToolMessage):
                    call = state["calls"].pop(m.id, {})
                    state["obs"].append({"step": len(state["obs"]) + 1, "tool": call.get("name", "?"),
                                         "arguments": call.get("arguments", {}), "content": _content(m.content),
                                         "status": "error" if m.error else "ok"})  # fmt: skip
                    state["turns"].append({"role": "assistant", "content": f"[{call.get('name', 'tool')} returned: "
                                                                           f"{str(m.content)[:2000]}]"})  # fmt: skip
                elif isinstance(m, UserMessage) and m.content:
                    state["turns"].append({"role": "user", "content": str(m.content)})
                elif getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                    state["turns"].append({"role": "assistant", "content": str(m.content)})

        def _context(self, state: dict[str, Any]) -> Context:
            turns = [t for t in state["turns"] if not t["content"].startswith("[")]  # tool results are observations
            return Context(messages=turns, observations=[Observation(**o) for o in state["obs"]], now=NOW,
                           tz="America/New_York", locale="en", trusted_tools=self.trusted)  # fmt: skip

        def generate_next_message(self, message: Any, state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            self._ingest(message, state)
            ctx = self._context(state)
            user_turn = isinstance(message, UserMessage)
            if state["pending"] is not None and user_turn:
                decision = self.router.resume(state["pending"], reply=str(message.content), context=ctx)
            else:
                decision = self.router.decide(ctx.messages, context=ctx, mode="loop")
            state["pending"] = None
            self.jev_cost += decision.usage.cost_usd or 0.0
            call = decision.call
            key = json.dumps([call.name, call.arguments], sort_keys=True, default=str) if call is not None else None
            self.decisions.append({"outcome": str(decision.outcome), "rule": decision.rule,
                                   "call": json.loads(key) if key else None})  # fmt: skip
            if (
                decision.outcome is Outcome.EXECUTE
                and call is not None
                and not (key == state["last"] and not user_turn)
            ):
                state["n"] += 1
                call_id = f"jev_{state['n']}"
                state["calls"][call_id] = {"name": call.name, "arguments": dict(call.arguments)}
                state["last"] = key
                return AssistantMessage(role="assistant", content=None, tool_calls=[
                    ToolCall(id=call_id, name=call.name, arguments=dict(call.arguments))]), state  # fmt: skip
            state["last"] = None
            if decision.outcome in (Outcome.CONFIRM, Outcome.CLARIFY) and decision.prompt and decision.prompt.text:
                state["pending"] = decision.pending
                text = decision.prompt.text
            else:
                text = self.writer.complete(state["turns"]) or "Is there anything else I can help you with?"
            state["turns"].append({"role": "assistant", "content": text})
            return AssistantMessage(role="assistant", content=text), state

    return JevtoolsAgent


def main(argv: Sequence[str] | None = None) -> int:
    """Run τ² tasks with the jevtools agent and/or τ²'s ``llm_agent`` (same LLM, same simulated customer)."""
    parser = argparse.ArgumentParser(prog="python -m jevtools.bench.tau2_agent", description=main.__doc__)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--tasks", type=int, default=20, help="N tasks of the domain, from --start")
    parser.add_argument("--start", type=int, default=0, help="index of the first task")
    parser.add_argument("--agent", choices=("jevtools", "llm", "both"), default="both")
    parser.add_argument("--llm", default="openrouter/openai/gpt-4.1-mini", help="the agent's LLM (litellm id)")
    parser.add_argument("--user-llm", default="openrouter/openai/gpt-4.1-mini", help="the simulated customer")
    parser.add_argument("--trust", choices=("none", "reads"), default="reads")
    parser.add_argument("--escalator", default=None, help="an OpenRouter model that drafts uncovered values")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out", required=True, help="JSON results")
    args = parser.parse_args(argv)

    from tau2.evaluator.evaluator import EvaluationType  # type: ignore[import-not-found]
    from tau2.orchestrator.orchestrator import Orchestrator  # type: ignore[import-not-found]
    from tau2.registry import registry  # type: ignore[import-not-found]
    from tau2.runner import build_environment, build_user, get_tasks, run_simulation  # type: ignore[import-not-found]

    from jevtools.backends import auto
    from jevtools.fallback import OpenAICompatibleEscalator

    tasks = get_tasks(args.domain)[args.start : args.start + args.tasks]
    results: dict[str, Any] = {"meta": vars(args), "runs": []}
    agents = ("jevtools", "llm") if args.agent == "both" else (args.agent,)
    jev = auto()
    for task in tasks:
        for kind in agents:
            env = build_environment(args.domain)
            if kind == "jevtools":
                escalator = (OpenAICompatibleEscalator(args.escalator, extra_body={"usage": {"include": True}})
                             if args.escalator else None)  # fmt: skip
                agent = agent_class()(env.get_tools(), env.get_policy(), backend=jev,
                                      llm=args.llm.removeprefix("openrouter/"), trust=args.trust,
                                      escalator=escalator)  # fmt: skip
            else:
                factory = registry.get_agent_factory("llm_agent")
                agent = factory(tools=env.get_tools(), domain_policy=env.get_policy(), llm=args.llm,
                                llm_args={"temperature": 0.0})  # fmt: skip
            user = build_user("user_simulator", env, task, llm=args.user_llm)
            orchestrator = Orchestrator(domain=args.domain, agent=agent, user=user, environment=env, task=task,
                                        max_steps=args.max_steps, max_errors=10, seed=42)  # fmt: skip
            try:
                sim = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
                reward = sim.reward_info.reward if sim.reward_info else None
                agent_cost = sim.agent_cost if getattr(sim, "agent_cost", None) is not None else None
                user_cost = getattr(sim, "user_cost", None)
                error = None
            except Exception as exc:  # noqa: BLE001 - a crashed simulation is a failed task, reported
                reward, agent_cost, user_cost, error = 0.0, None, None, f"{type(exc).__name__}: {exc}"[:300]
            row = {"task": str(task.id), "agent": kind, "reward": reward, "agent_cost": agent_cost,
                   "user_cost": user_cost, "error": error}  # fmt: skip
            if kind == "jevtools":
                row.update(jev_cost=agent.jev_cost, writer_cost=agent.writer.cost_usd,
                           drafter_cost=getattr(agent.router.escalator, "cost_usd", 0.0) if agent.router.escalator
                           else 0.0, decisions=agent.decisions)  # fmt: skip
            results["runs"].append(row)
            print(f"{args.domain} task {task.id:>4} {kind:<8} reward {reward}" + (f"  ERROR {error}" if error else ""),
                  file=sys.stderr, flush=True)  # fmt: skip
            Path(args.out).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    for kind in agents:
        rows = [r for r in results["runs"] if r["agent"] == kind]
        mean = sum(float(r["reward"] or 0) for r in rows) / len(rows) if rows else 0.0
        print(f"{kind}: pass^1 {mean:.1%} over {len(rows)} task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

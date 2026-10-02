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


BLOCKED = (
    "Not executed: the {tool} call is not supported by the conversation ({reason}). Look the missing value up with a "
    "tool or ask the customer, then try again."
)


def guard_agent_class() -> Any:
    """The ``GuardAgent`` class: τ²'s own LLM agent plans and proposes every call; read calls pass through; each other
    call is checked by jevtools (:meth:`jevtools.router.Router.check`) before it runs. ``execute`` runs it with
    jevtools' bound arguments, ``confirm`` shows jevtools' card (the customer's reply resumes it), anything else
    returns a tool error to the LLM, which re-plans (twice at most, then it must answer in text)."""
    from tau2.agent.llm_agent import LLMAgent  # type: ignore[import-not-found]
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.utils.llm_utils import generate  # type: ignore[import-not-found]

    class GuardAgent(LLMAgent):  # type: ignore[misc]
        def __init__(self, tools: Sequence[Any], domain_policy: str, *, backend: Any, llm: str, trust: str = "reads",
                     llm_args: dict[str, Any] | None = None, verify: bool = True) -> None:  # fmt: skip
            super().__init__(tools=list(tools), domain_policy=domain_policy, llm=llm, llm_args=llm_args or {})
            self.verify = verify
            schemas = [deepcopy(t.openai_schema) for t in tools]
            names = [s["function"]["name"] for s in schemas]
            self.reads = {n for n in names if n.startswith(READ_PREFIXES)}
            self.trusted = tuple(self.reads) if trust == "reads" else ()
            self.router = Router(schemas, backend=backend)
            self.pending: Any = None
            self.jev_cost = 0.0
            self.checks: list[dict[str, Any]] = []

        def _context(self, state: Any) -> Context:
            turns: list[dict[str, str]] = []
            obs: list[Observation] = []
            calls: dict[str, Any] = {}
            for m in state.messages:
                if isinstance(m, ToolMessage):
                    call = calls.pop(m.id, None)
                    obs.append(Observation(step=len(obs) + 1, tool=call.name if call else "?",
                                           arguments=dict(call.arguments) if call else {}, content=_content(m.content),
                                           status="error" if m.error else "ok"))  # fmt: skip
                    continue
                for tc in getattr(m, "tool_calls", None) or []:
                    calls[tc.id] = tc
                if getattr(m, "role", None) in ("user", "assistant") and getattr(m, "content", None):
                    turns.append({"role": str(m.role), "content": str(m.content)})
            return Context(messages=turns, observations=obs, now=NOW, tz="America/New_York", locale="en",
                           trusted_tools=self.trusted)  # fmt: skip

        def _llm(self, state: Any, *, tools: bool = True) -> Any:
            return generate(model=self.llm, tools=self.tools if tools else None,
                            messages=state.system_messages + state.messages, call_name="agent_response",
                            **self.llm_args)  # fmt: skip

        def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            if self.pending is not None and isinstance(message, UserMessage):
                ctx = self._context(state)
                decision = self.router.resume(self.pending, reply=str(message.content), context=ctx)
                self.pending = None
                self.jev_cost += decision.usage.cost_usd or 0.0
                reply = self._act(decision, None, state)
                if reply is not None:
                    return reply, state
            for _ in range(3):
                proposal = self._llm(state)
                writes = [tc for tc in proposal.tool_calls or [] if tc.name not in self.reads]
                if not writes:
                    state.messages.append(proposal)
                    return proposal, state
                tc = writes[0]
                ctx = self._context(state)
                decision = self.router.check(ctx.messages, {"name": tc.name, "arguments": dict(tc.arguments)},
                                             context=ctx, mode="loop", verify=self.verify)  # fmt: skip
                self.jev_cost += decision.usage.cost_usd or 0.0
                self.checks.append({"tool": tc.name, "outcome": str(decision.outcome), "rule": decision.rule,
                                    "proposed": dict(tc.arguments),
                                    "bound": dict(decision.call.arguments) if decision.call else None,
                                    "bottleneck": decision.bottleneck.slot if decision.bottleneck else None,
                                    "slot_p": {k: v.p for k, v in decision.slots.items()},
                                    "C": decision.confidence.call if decision.confidence else None,
                                    "notes": [n for n in (decision.trace.notes if decision.trace else [])
                                              if n.startswith("check:")]})  # fmt: skip
                reply = self._act(decision, tc, state)
                if reply is not None:
                    return reply, state
                state.messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
                slot = decision.bottleneck.slot if decision.bottleneck is not None else None
                if decision.rule.startswith("C1") and slot:
                    reason = f"the value for '{slot}' was neither given by the customer nor returned by a lookup tool"
                else:
                    reason = decision.prompt.text if decision.prompt and decision.prompt.text else decision.rule
                state.messages.append(ToolMessage(id=tc.id, role="tool", content=BLOCKED.format(tool=tc.name,
                                                  reason=reason), requestor="assistant", error=True))  # fmt: skip
            final = self._llm(state, tools=False)
            state.messages.append(final)
            return final, state

        def _act(self, decision: Any, tc: Any, state: Any) -> Any:
            """The message for an ``execute`` (the bound call) or a ``confirm`` (the card); ``None`` to block."""
            if decision.outcome is Outcome.EXECUTE and decision.call is not None:
                call_id = tc.id if tc is not None else f"jev_{len(self.checks)}"
                bound = ToolCall(id=call_id, name=decision.call.name, arguments=dict(decision.call.arguments))
                out = AssistantMessage(role="assistant", content=None, tool_calls=[bound])
                state.messages.append(out)
                return out
            if decision.outcome is Outcome.CONFIRM and decision.prompt and decision.prompt.text:
                self.pending = decision.pending
                out = AssistantMessage(role="assistant", content=decision.prompt.text)
                state.messages.append(out)
                return out
            return None

    return GuardAgent


def main(argv: Sequence[str] | None = None) -> int:
    """Run τ² tasks with the jevtools agent and/or τ²'s ``llm_agent`` (same LLM, same simulated customer)."""
    parser = argparse.ArgumentParser(prog="python -m jevtools.bench.tau2_agent", description=main.__doc__)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--tasks", type=int, default=20, help="N tasks of the domain, from --start")
    parser.add_argument("--start", type=int, default=0, help="index of the first task")
    parser.add_argument("--agent", choices=("jevtools", "llm", "guard", "both"), default="both",
                        help="jevtools decides every call; llm: τ²'s llm_agent; guard: the LLM plans, jevtools checks "
                             "every non-read call; both: jevtools and llm")  # fmt: skip
    parser.add_argument("--llm", default="openrouter/openai/gpt-4.1-mini", help="the agent's LLM (litellm id)")
    parser.add_argument("--user-llm", default="openrouter/openai/gpt-4.1-mini", help="the simulated customer")
    parser.add_argument("--trust", choices=("none", "reads"), default="reads")
    parser.add_argument("--escalator", default=None, help="an OpenRouter model that drafts uncovered values")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--guard-verify",
        choices=("on", "off"),
        default="on",
        help="guard: also have Jev verify grounded values (off: grounding in code only)",
    )
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
            if kind == "guard":
                agent = guard_agent_class()(env.get_tools(), env.get_policy(), backend=jev, llm=args.llm,
                                            trust=args.trust, llm_args={"temperature": 0.0},
                                            verify=args.guard_verify == "on")  # fmt: skip
            elif kind == "jevtools":
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
            if kind == "guard":
                row.update(jev_cost=agent.jev_cost, checks=agent.checks)
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

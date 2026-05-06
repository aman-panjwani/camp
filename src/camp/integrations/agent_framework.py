"""
camp.integrations.agent_framework
----------------------------------
Microsoft Agent Framework integration for CAMP PII protection.

Requires: pip install campii[agent-framework]

CAMP sits as AgentMiddleware at the user → agent boundary:
  - Intercepts context.messages[-1] (the current user turn)
  - Applies CAMP masking: PASS / PSEUDONYMIZE / BLOCK
  - After agent runs, demaskes context.result before it reaches the user

Reference: https://learn.microsoft.com/en-us/agent-framework/agents/middleware/

Usage - class-based (recommended, stateful across runs):
    from camp.integrations.agent_framework import CAMPAgentMiddleware

    async with Agent(
        client=FoundryChatClient(credential=credential),
        name="MyAgent",
        instructions="You are a helpful assistant.",
        middleware=[CAMPAgentMiddleware(threshold=2.0, alpha=0.3)],
    ) as agent:
        result = await agent.run("My name is Sarah Johnson, SSN 512-34-7891")
        # Real PII never reached the LLM

Usage - function-based factory (lightweight, per-run):
    from camp.integrations.agent_framework import create_camp_middleware

    camp = create_camp_middleware(threshold=2.0)
    result = await agent.run("Hello", middleware=[camp])
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from camp.core.masker import BLOCK, CAMPMasker, TurnResult

try:
    from agent_framework import (
        AgentContext,
        AgentMiddleware,
        AgentResponse,
        Message,
        MiddlewareTermination,
        agent_middleware,
    )
except ImportError as exc:
    raise ImportError(
        "Microsoft Agent Framework integration requires agent-framework. "
        "Install with: pip install campii[agent-framework]"
    ) from exc


class CAMPAgentMiddleware(AgentMiddleware):
    """
    Microsoft Agent Framework middleware applying CAMP PII protection.

    Sits between the user and the agent at every run. Intercepts the
    latest user message, applies CAMP (PASS / PSEUDONYMIZE / BLOCK),
    and demaskes the agent response before it returns to the user.

    Register at agent level (all runs) or run level (single run):

        # All runs
        async with Agent(..., middleware=[CAMPAgentMiddleware()]) as agent:
            result = await agent.run("My SSN is 512-34-7891")

        # Single run only
        result = await agent.run("...", middleware=[CAMPAgentMiddleware()])
    """

    def __init__(
        self,
        threshold:   float           = 2.0,
        alpha:       float           = 0.3,
        session_id:  str             = "default",
        redaction_map: dict[str, str] | None = None,
    ) -> None:
        self._threshold   = threshold
        self._alpha       = alpha
        self._session_id  = session_id
        self._redaction_map = redaction_map
        self._masker      = CAMPMasker(
            threshold=threshold, alpha=alpha,
            session_id=session_id, redaction_map=redaction_map,
        )
        self._turn_index = 0
        self._last_result: TurnResult | None = None

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        # ── Intercept the latest user message ────────────────
        if context.messages:
            last_message = context.messages[-1]
            raw_text     = last_message.text or ""

            result              = self._masker.process_turn(raw_text, self._turn_index)
            self._turn_index   += 1
            self._last_result   = result

            # Message.text is a read-only computed property - replace the message
            context.messages[-1] = Message(
                role=last_message.role,
                contents=[result.sent_to_llm],
            )

            # Hard block: terminate before the agent runs
            if result.decision == BLOCK and result.sent_to_llm != raw_text:
                context.result = AgentResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=[
                                "This message contains sensitive information "
                                "that cannot be forwarded to the agent."
                            ],
                        )
                    ]
                )
                raise MiddlewareTermination(result=context.result)

        # ── Continue to agent / next middleware ───────────────
        await call_next()

        # ── Demask response before returning to user ──────────
        if context.result:
            for i, msg in enumerate(context.result.messages):
                if msg.text:
                    demasked = self._masker.demask_response(msg.text)
                    context.result.messages[i] = Message(role=msg.role, contents=[demasked])

    # ── Inspection ────────────────────────────────────────────

    @property
    def cpe_score(self) -> float:
        history = self._masker.cpe_history()
        return history[-1] if history else 0.0

    @property
    def triggered(self) -> bool:
        return self._masker.scorer.triggered()

    @property
    def last_result(self) -> TurnResult | None:
        return self._last_result

    @property
    def pseudonym_map(self) -> dict:
        return self._masker.pseudonym_map()

    @property
    def masker(self) -> CAMPMasker:
        return self._masker

    def reset(self) -> None:
        """Start a fresh CAMP session with the same configuration."""
        self._masker = CAMPMasker(
            threshold=self._threshold,
            alpha=self._alpha,
            session_id=self._session_id,
            redaction_map=self._redaction_map,
        )
        self._turn_index  = 0
        self._last_result = None


def create_camp_middleware(
    threshold:   float           = 2.0,
    alpha:       float           = 0.3,
    session_id:  str             = "default",
    redaction_map: dict[str, str] | None = None,
) -> Any:
    """
    Factory for a function-based CAMP agent middleware.
    Lightweight alternative for simple or per-run use cases.

    Usage:
        camp   = create_camp_middleware(threshold=1.5)
        result = await agent.run("My name is Sarah", middleware=[camp])
    """
    masker = CAMPMasker(
        threshold=threshold, alpha=alpha, session_id=session_id, redaction_map=redaction_map
    )
    turn_index_box = [0]  # mutable container for closure state

    @agent_middleware
    async def _camp_middleware(
        context:   AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if context.messages:
            last_message = context.messages[-1]
            raw_text     = last_message.text or ""

            result             = masker.process_turn(raw_text, turn_index_box[0])
            turn_index_box[0] += 1
            context.messages[-1] = Message(
                role=last_message.role,
                contents=[result.sent_to_llm],
            )

            if result.decision == BLOCK and result.sent_to_llm != raw_text:
                context.result = AgentResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=["Blocked: sensitive information detected."],
                        )
                    ]
                )
                raise MiddlewareTermination(result=context.result)

        await call_next()

        if context.result:
            for i, msg in enumerate(context.result.messages):
                if msg.text:
                    demasked = masker.demask_response(msg.text)
                    context.result.messages[i] = Message(role=msg.role, contents=[demasked])

    return _camp_middleware

"""
Microsoft Agent Framework integration tests.
Stubs out agent_framework so the package is not required in CI.
"""
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Stub out agent_framework before importing the integration ─────

def _make_af_stub():
    af = ModuleType("agent_framework")

    class AgentMiddleware:
        async def process(self, context, call_next): ...

    class AgentContext:
        def __init__(self, text=""):
            msg      = MagicMock()
            msg.text = text
            self.messages = [msg]
            self.result   = None

    class AgentResponse:
        def __init__(self, messages=None):
            self.messages = messages or []

    class Message:
        def __init__(self, role="assistant", contents=None):
            self.role     = role
            self.contents = contents or []
            self.text     = " ".join(contents) if contents else ""

    class MiddlewareTermination(Exception):
        def __init__(self, result=None):
            self.result = result

    def agent_middleware(fn):
        return fn

    af.AgentMiddleware      = AgentMiddleware
    af.AgentContext         = AgentContext
    af.AgentResponse        = AgentResponse
    af.Message              = Message
    af.MiddlewareTermination = MiddlewareTermination
    af.agent_middleware     = agent_middleware

    sys.modules.setdefault("agent_framework", af)
    return af, AgentContext, AgentResponse, Message, MiddlewareTermination


_af, AgentContext, AgentResponse, Message, MiddlewareTermination = _make_af_stub()

from camp.integrations.agent_framework import (  # noqa: E402
    CAMPAgentMiddleware,
    create_camp_middleware,
)

# ── Helpers ───────────────────────────────────────────────────────

def _ctx(text: str) -> AgentContext:
    return AgentContext(text)


# ── CAMPAgentMiddleware properties ────────────────────────────────

def test_initial_cpe_score_is_zero():
    mw = CAMPAgentMiddleware()
    assert mw.cpe_score == 0.0


def test_initial_triggered_is_false():
    mw = CAMPAgentMiddleware()
    assert not mw.triggered


def test_initial_last_result_is_none():
    mw = CAMPAgentMiddleware()
    assert mw.last_result is None


def test_pseudonym_map_starts_empty():
    mw = CAMPAgentMiddleware()
    assert mw.pseudonym_map == {}


# ── process - PASS ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_passes_clean_message():
    mw        = CAMPAgentMiddleware(threshold=5.0)
    ctx       = _ctx("Hello, how are you?")
    call_next = AsyncMock()
    await mw.process(ctx, call_next)
    call_next.assert_awaited_once()
    assert ctx.messages[-1].text == "Hello, how are you?"


@pytest.mark.asyncio
async def test_process_call_next_called():
    mw        = CAMPAgentMiddleware(threshold=5.0)
    ctx       = _ctx("Hello")
    call_next = AsyncMock()
    await mw.process(ctx, call_next)
    assert call_next.await_count == 1


# ── process - BLOCK ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_blocks_ssn():
    mw        = CAMPAgentMiddleware(threshold=5.0)
    ctx       = _ctx("My SSN is 512-34-7891.")
    call_next = AsyncMock()
    with pytest.raises(MiddlewareTermination):
        await mw.process(ctx, call_next)
    assert "512-34-7891" not in ctx.messages[-1].text


@pytest.mark.asyncio
async def test_block_does_not_call_next():
    mw        = CAMPAgentMiddleware(threshold=5.0)
    ctx       = _ctx("My SSN is 512-34-7891.")
    call_next = AsyncMock()
    with pytest.raises(MiddlewareTermination):
        await mw.process(ctx, call_next)
    call_next.assert_not_awaited()


# ── process - demask ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_demaskes_result():
    mw = CAMPAgentMiddleware(threshold=5.0)
    ctx = _ctx("Hello")
    response_msg      = MagicMock()
    response_msg.text = "Hello back."
    ctx_result        = MagicMock()
    ctx_result.messages = [response_msg]

    async def _set_result():
        ctx.result = ctx_result

    await mw.process(ctx, _set_result)
    assert isinstance(response_msg.text, str)


# ── last_result updated ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_last_result_set_after_process():
    mw        = CAMPAgentMiddleware(threshold=5.0)
    ctx       = _ctx("My name is Alice.")
    call_next = AsyncMock()
    await mw.process(ctx, call_next)
    assert mw.last_result is not None
    assert mw.last_result.turn_index == 0


# ── reset ─────────────────────────────────────────────────────────

def test_reset_clears_state():
    mw = CAMPAgentMiddleware(threshold=2.0)
    mw.reset()
    assert mw.cpe_score == 0.0
    assert not mw.triggered
    assert mw.last_result is None


def test_reset_preserves_config():
    mw = CAMPAgentMiddleware(threshold=1.5, alpha=0.5)
    mw.reset()
    assert mw._threshold == 1.5
    assert mw._alpha == 0.5


# ── create_camp_middleware ────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_camp_middleware_returns_callable():
    camp = create_camp_middleware(threshold=5.0)
    assert callable(camp)


@pytest.mark.asyncio
async def test_function_middleware_passes_clean():
    camp      = create_camp_middleware(threshold=5.0)
    ctx       = _ctx("Hello")
    call_next = AsyncMock()
    await camp(ctx, call_next)
    call_next.assert_awaited_once()

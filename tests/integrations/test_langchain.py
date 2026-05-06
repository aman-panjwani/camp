"""
LangChain integration tests - mock langchain_core to avoid the dependency in CI.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

# ── Stub out langchain_core before importing the integration ──────

def _make_langchain_stub():
    lc        = ModuleType("langchain_core")
    callbacks = ModuleType("langchain_core.callbacks")
    messages  = ModuleType("langchain_core.messages")
    outputs   = ModuleType("langchain_core.outputs")

    class BaseCallbackHandler:
        raise_error = False
        def __init_subclass__(cls, **kwargs): pass

    class BaseMessage:
        def __init__(self, content=""): self.content = content

    class Generation:
        def __init__(self, text=""): self.text = text

    class LLMResult:
        def __init__(self, generations=None): self.generations = generations or []

    callbacks.BaseCallbackHandler = BaseCallbackHandler
    messages.BaseMessage          = BaseMessage
    outputs.LLMResult             = LLMResult

    lc.callbacks = callbacks
    lc.messages  = messages
    lc.outputs   = outputs

    sys.modules.setdefault("langchain_core",           lc)
    sys.modules.setdefault("langchain_core.callbacks", callbacks)
    sys.modules.setdefault("langchain_core.messages",  messages)
    sys.modules.setdefault("langchain_core.outputs",   outputs)

    return lc, Generation, LLMResult

_lc_stub, Generation, LLMResult = _make_langchain_stub()

from camp.integrations.langchain import CAMPCallbackHandler, CAMPChain  # noqa: E402

# ── CAMPCallbackHandler ───────────────────────────────────────────

def test_handler_initial_state():
    handler = CAMPCallbackHandler(threshold=2.0)
    assert handler.cpe_score == 0.0
    assert not handler.triggered
    assert handler.last_result is None


def test_on_llm_start_mutates_prompts_in_place():
    handler = CAMPCallbackHandler(threshold=5.0)
    prompts = ["Hello, this is a test."]
    handler.on_llm_start({}, prompts, run_id=uuid4())
    # Clean message → unchanged content (PASS)
    assert prompts[0] == "Hello, this is a test."


def test_on_llm_start_blocks_ssn():
    handler = CAMPCallbackHandler(threshold=5.0)
    prompts = ["My SSN is 512-34-7891."]
    handler.on_llm_start({}, prompts, run_id=uuid4())
    assert "512-34-7891" not in prompts[0]
    assert "[BLOCKED]" in prompts[0]


def test_on_llm_start_sets_last_result():
    handler = CAMPCallbackHandler(threshold=5.0)
    handler.on_llm_start({}, ["Hello"], run_id=uuid4())
    assert handler.last_result is not None
    assert handler.last_result.turn_index == 0


def test_on_llm_end_calls_demask():
    handler    = CAMPCallbackHandler(threshold=5.0)
    generation = Generation(text="Response text.")
    response   = LLMResult(generations=[[generation]])
    handler.on_llm_end(response, run_id=uuid4())
    assert isinstance(generation.text, str)


def test_on_chat_model_start_masks_last_message():
    from langchain_core.messages import BaseMessage
    handler = CAMPCallbackHandler(threshold=5.0)
    msg     = BaseMessage("My SSN is 512-34-7891.")
    handler.on_chat_model_start({}, [[msg]], run_id=uuid4())
    assert "512-34-7891" not in msg.content


def test_handler_turn_counter_increments():
    handler = CAMPCallbackHandler(threshold=5.0)
    handler.on_llm_start({}, ["Turn 1"], run_id=uuid4())
    handler.on_llm_start({}, ["Turn 2"], run_id=uuid4())
    assert handler.last_result.turn_index == 1


# ── CAMPChain ─────────────────────────────────────────────────────

def test_camp_chain_from_runnable():
    runnable = MagicMock()
    runnable.invoke.return_value = {"output": "hello"}
    chain  = CAMPChain.from_runnable(runnable, threshold=2.0)
    result = chain.invoke({"input": "Hello"})
    assert runnable.invoke.called
    assert result == {"output": "hello"}


def test_camp_chain_injects_handler_as_callback():
    runnable = MagicMock()
    runnable.invoke.return_value = {}
    chain = CAMPChain.from_runnable(runnable)
    chain.invoke({"input": "Hello"})
    call_kwargs = runnable.invoke.call_args[1]
    assert chain.handler in call_kwargs["callbacks"]


def test_camp_chain_handler_property():
    runnable = MagicMock()
    chain    = CAMPChain.from_runnable(runnable)
    assert isinstance(chain.handler, CAMPCallbackHandler)

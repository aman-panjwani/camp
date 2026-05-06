import pytest

from camp.core.masker import BLOCK, PASS
from camp.integrations.llm import CAMPSession


def _echo(message: str) -> str:
    return f"Echo: {message}"


# ── CAMPSession.process ───────────────────────────────────────────

def test_process_returns_turn_result(camp_session: CAMPSession):
    result = camp_session.process("Hello, how are you?")
    assert result.decision == PASS
    assert result.sent_to_llm == "Hello, how are you?"


def test_process_increments_turn_count(camp_session: CAMPSession):
    camp_session.process("Turn 1")
    camp_session.process("Turn 2")
    assert camp_session.turn_count == 2


def test_process_blocks_ssn(camp_session: CAMPSession):
    result = camp_session.process("My SSN is 512-34-7891.")
    assert result.decision == BLOCK
    assert "512-34-7891" not in result.sent_to_llm


def test_cpe_score_zero_initially(camp_session: CAMPSession):
    assert camp_session.cpe_score == 0.0


def test_cpe_score_after_turn(camp_session: CAMPSession):
    camp_session.process("My name is John Smith.")
    assert camp_session.cpe_score >= 0.0


def test_triggered_false_initially(camp_session: CAMPSession):
    assert not camp_session.triggered


# ── CAMPSession.wrap ──────────────────────────────────────────────

def test_wrap_creates_bound_session():
    session = CAMPSession.wrap(_echo, threshold=2.0)
    assert session._llm_fn is _echo


def test_chat_calls_llm_with_masked_text():
    received: list[str] = []

    def capturing_llm(msg: str) -> str:
        received.append(msg)
        return "OK"

    session = CAMPSession.wrap(capturing_llm, threshold=0.1)
    session.chat("My SSN is 512-34-7891.")
    assert len(received) == 1
    assert "512-34-7891" not in received[0]


def test_chat_demaskes_response():
    def named_llm(msg: str) -> str:
        return "I'll help you."

    session = CAMPSession.wrap(named_llm, threshold=2.0)
    response = session.chat("Hello there")
    assert isinstance(response, str)


def test_chat_without_llm_raises():
    session = CAMPSession(threshold=2.0)
    with pytest.raises(RuntimeError, match="No LLM callable bound"):
        session.chat("Hello")


# ── CAMPSession.demask ────────────────────────────────────────────

def test_demask_returns_string(camp_session: CAMPSession):
    result = camp_session.demask("Some response text.")
    assert isinstance(result, str)


# ── CAMPSession.reset ─────────────────────────────────────────────

def test_reset_clears_turn_count(camp_session: CAMPSession):
    camp_session.process("Turn 1")
    camp_session.process("Turn 2")
    camp_session.reset()
    assert camp_session.turn_count == 0


def test_reset_clears_cpe_score(camp_session: CAMPSession):
    camp_session.process("My name is John Smith.")
    camp_session.reset()
    assert camp_session.cpe_score == 0.0


def test_reset_preserves_config():
    session = CAMPSession(threshold=1.5, alpha=0.5)
    session.reset()
    assert session._threshold == 1.5
    assert session._alpha == 0.5

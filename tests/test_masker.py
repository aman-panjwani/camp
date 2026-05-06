from camp.core.masker import BLOCK, PASS, PSEUDONYMIZE, CAMPMasker, TurnResult

# ── Initial state ─────────────────────────────────────────────────

def test_initial_state(masker: CAMPMasker):
    assert masker.threshold == 2.0
    assert masker.cpe_history() == []
    assert masker.trigger_turn() is None
    assert masker.pseudonym_map() == {}
    assert masker.results() == []


# ── TurnResult structure ──────────────────────────────────────────

def test_turn_result_is_dataclass(masker: CAMPMasker):
    result = masker.process_turn("Hello", turn_index=0)
    assert isinstance(result, TurnResult)
    assert result.turn_index == 0
    assert result.raw_text == "Hello"
    assert isinstance(result.cpe_score, float)
    assert result.risk_band in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert result.decision in {PASS, PSEUDONYMIZE, BLOCK}


# ── PASS decision ─────────────────────────────────────────────────

def test_pass_decision_clean_message(masker: CAMPMasker):
    result = masker.process_turn("Hello, how are you?", turn_index=0)
    assert result.decision == PASS
    assert result.sent_to_llm == "Hello, how are you?"
    assert not result.triggered


# ── BLOCK decision ────────────────────────────────────────────────

def test_block_decision_on_ssn(masker: CAMPMasker):
    result = masker.process_turn("My SSN is 512-34-7891.", turn_index=0)
    assert result.decision == BLOCK
    assert "512-34-7891" not in result.sent_to_llm
    assert "[BLOCKED]" in result.sent_to_llm


def test_block_does_not_reach_llm(masker: CAMPMasker):
    result = masker.process_turn("My SSN is 512-34-7891.", turn_index=0)
    assert "512-34-7891" not in result.sent_to_llm


# ── CPE history ───────────────────────────────────────────────────

def test_cpe_history_grows_with_turns(masker: CAMPMasker):
    for i in range(4):
        masker.process_turn(f"Turn {i}", turn_index=i)
    assert len(masker.cpe_history()) == 4


def test_cpe_history_non_negative(masker: CAMPMasker):
    for i in range(3):
        masker.process_turn(f"Message {i}", turn_index=i)
    assert all(s >= 0.0 for s in masker.cpe_history())


# ── PSEUDONYMIZE decision ─────────────────────────────────────────

def test_pseudonymize_triggered_at_low_threshold(low_threshold_masker: CAMPMasker):
    turns = [
        "My name is Sarah Johnson.",
        "I live in Denver, Colorado.",
        "I work at Lockheed Martin.",
    ]
    results = [low_threshold_masker.process_turn(t, i) for i, t in enumerate(turns)]
    decisions = {r.decision for r in results}
    assert PSEUDONYMIZE in decisions


def test_pseudonymized_history_hides_real_name(low_threshold_masker: CAMPMasker):
    low_threshold_masker.process_turn("My name is Sarah Johnson.", 0)
    result = low_threshold_masker.process_turn("I live in Denver, Colorado.", 1)
    if result.decision == PSEUDONYMIZE and result.rewritten_history:
        for turn_text in result.rewritten_history:
            assert "Sarah Johnson" not in turn_text


# ── Results list ──────────────────────────────────────────────────

def test_results_accumulate(masker: CAMPMasker):
    for i in range(3):
        masker.process_turn(f"Message {i}", turn_index=i)
    assert len(masker.results()) == 3


# ── Demask ────────────────────────────────────────────────────────

def test_demask_response_returns_string(masker: CAMPMasker):
    masker.process_turn("My name is Sarah Johnson.", turn_index=0)
    result = masker.demask_response("Hello there.")
    assert isinstance(result, str)

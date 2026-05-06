"""
Extractor tests use mocked Presidio to avoid loading the spaCy model in CI.
Integration tests that require the real model are marked with @pytest.mark.slow.
"""


from camp.core.entities import ACCOUNT, LOCATION, PERSON, SSN
from camp.core.extractor import DetectedEntity, mask_text


def _entity(etype: str, value: str, start: int = 0) -> DetectedEntity:
    return DetectedEntity(entity_type=etype, value=value, score=0.9, turn_index=0)


# ── DetectedEntity behaviour ──────────────────────────────────────

def test_entity_weight(person_entity: DetectedEntity):
    assert 0 < person_entity.weight() <= 1.0


def test_entity_label(person_entity: DetectedEntity):
    assert isinstance(person_entity.label(), str)
    assert len(person_entity.label()) > 0


def test_entity_is_hard_block_false_for_person(person_entity: DetectedEntity):
    assert not person_entity.is_hard_block()


def test_entity_is_hard_block_true_for_ssn(ssn_entity: DetectedEntity):
    assert ssn_entity.is_hard_block()


def test_entity_repr(person_entity: DetectedEntity):
    r = repr(person_entity)
    assert "Person Name" in r
    assert "Michael Torres" in r


# ── mask_text ─────────────────────────────────────────────────────

def test_mask_text_no_entities():
    text = "Hello, how are you?"
    assert mask_text(text, []) == text


def test_mask_text_person_placeholder():
    entities = [_entity(PERSON, "Sarah Johnson")]
    result   = mask_text("My name is Sarah Johnson.", entities)
    assert "Sarah Johnson" not in result
    assert "[Person Name]" in result


def test_mask_text_ssn_becomes_blocked():
    entities = [_entity(SSN, "512-34-7891")]
    result   = mask_text("My SSN is 512-34-7891.", entities)
    assert "512-34-7891" not in result
    assert "[BLOCKED]" in result


def test_mask_text_account_becomes_blocked():
    entities = [_entity(ACCOUNT, "account ending in 4872")]
    result   = mask_text("I bank with account ending in 4872.", entities)
    assert "4872" not in result
    assert "[BLOCKED]" in result


def test_mask_text_case_insensitive():
    entities = [_entity(PERSON, "Sarah Johnson")]
    result   = mask_text("SARAH JOHNSON called.", entities)
    assert "SARAH JOHNSON" not in result


def test_mask_text_multiple_entities():
    entities = [
        _entity(PERSON, "Alice"),
        _entity(LOCATION, "New York"),
    ]
    text   = "Alice lives in New York."
    result = mask_text(text, entities)
    assert "Alice" not in result
    assert "New York" not in result

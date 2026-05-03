from camp.core.pseudonymizer import Pseudonymizer
from camp.core.extractor import DetectedEntity
from camp.core.entities import (
    PERSON, LOCATION, ORGANIZATION, EMAIL, PHONE,
    SSN, CREDIT_CARD, ACCOUNT, SALARY, AGE,
    FINANCIAL_AMOUNT, FINANCIAL_METRIC,
    INTERNAL_PROJECTION, CONFIDENTIAL_DATA,
)


def _entity(etype: str, value: str) -> DetectedEntity:
    return DetectedEntity(entity_type=etype, value=value, score=0.9, turn_index=0)


# ── Hard block ────────────────────────────────────────────────────

def test_ssn_returns_blocked():
    p = Pseudonymizer()
    assert p.get_pseudonym(_entity(SSN, "512-34-7891")) == "[BLOCKED]"


def test_credit_card_returns_blocked():
    p = Pseudonymizer()
    assert p.get_pseudonym(_entity(CREDIT_CARD, "4111111111111111")) == "[BLOCKED]"


def test_account_returns_blocked():
    p = Pseudonymizer()
    assert p.get_pseudonym(_entity(ACCOUNT, "account ending in 4872")) == "[BLOCKED]"


# ── Consistency ───────────────────────────────────────────────────

def test_same_value_same_pseudonym():
    p = Pseudonymizer()
    e1 = _entity(PERSON, "Sarah Johnson")
    e2 = _entity(PERSON, "Sarah Johnson")
    assert p.get_pseudonym(e1) == p.get_pseudonym(e2)


def test_different_values_different_pseudonyms():
    p = Pseudonymizer()
    assert p.get_pseudonym(_entity(PERSON, "Alice Smith")) != p.get_pseudonym(_entity(PERSON, "Bob Jones"))


def test_person_generates_non_empty_name():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(PERSON, "Michael Torres"))
    assert pseudo != "Michael Torres"
    assert len(pseudo) > 0


# ── Generation for each type ──────────────────────────────────────

def test_location_generates_fake():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(LOCATION, "Austin, Texas"))
    assert isinstance(pseudo, str) and len(pseudo) > 0
    assert pseudo != "Austin, Texas"


def test_email_generates_fake():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(EMAIL, "sarah@example.com"))
    assert "@" in pseudo


def test_salary_generates_number():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(SALARY, "92,000 dollars a year"))
    assert "dollars" in pseudo.lower() or any(c.isdigit() for c in pseudo)


def test_financial_amount_generates_dollar_value():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(FINANCIAL_AMOUNT, "$340M"))
    assert "$" in pseudo


def test_confidential_data_neutralizes():
    p      = Pseudonymizer()
    pseudo = p.get_pseudonym(_entity(CONFIDENTIAL_DATA, "not been disclosed externally yet"))
    assert pseudo != "not been disclosed externally yet"
    assert len(pseudo) > 0


# ── Text substitution ─────────────────────────────────────────────

def test_pseudonymize_text_removes_real_name():
    p        = Pseudonymizer()
    entities = [_entity(PERSON, "Sarah Johnson")]
    text     = "Hello my name is Sarah Johnson, how can I help?"
    result   = p.pseudonymize_text(text, entities)
    assert "Sarah Johnson" not in result


def test_pseudonymize_text_no_entities_unchanged():
    p      = Pseudonymizer()
    text   = "Hello, how are you?"
    result = p.pseudonymize_text(text, [])
    assert result == text


# ── De-masking ────────────────────────────────────────────────────

def test_demask_response_restores_name():
    p      = Pseudonymizer()
    entity = _entity(PERSON, "Sarah Johnson")
    pseudo = p.get_pseudonym(entity)

    fake_response = f"I can help you, {pseudo}."
    demasked      = p.demask_response(fake_response)
    assert "Sarah Johnson" in demasked


def test_demask_no_pseudonyms_unchanged():
    p      = Pseudonymizer()
    result = p.demask_response("Nothing to restore here.")
    assert result == "Nothing to restore here."


# ── Map inspection ────────────────────────────────────────────────

def test_pseudonym_map_populated():
    p      = Pseudonymizer()
    entity = _entity(PERSON, "Alice")
    p.get_pseudonym(entity)
    assert "Alice" in p.pseudonym_map()


def test_reverse_map_populated():
    p      = Pseudonymizer()
    entity = _entity(PERSON, "Alice")
    pseudo = p.get_pseudonym(entity)
    assert pseudo in p.reverse_map()
    assert p.reverse_map()[pseudo] == "Alice"

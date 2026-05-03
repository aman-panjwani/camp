from camp.core.entities import (
    ALL_ENTITY_TYPES,
    ENTITY_LABELS,
    ENTITY_WEIGHTS,
    HARD_BLOCK_TYPES,
    SSN,
    CREDIT_CARD,
    ACCOUNT,
    PERSON,
    get_risk_band,
    get_risk_color,
)


def test_all_entity_types_have_weights():
    for etype in ALL_ENTITY_TYPES:
        assert etype in ENTITY_WEIGHTS, f"{etype} missing from ENTITY_WEIGHTS"


def test_all_entity_types_have_labels():
    for etype in ALL_ENTITY_TYPES:
        assert etype in ENTITY_LABELS, f"{etype} missing from ENTITY_LABELS"


def test_weights_in_valid_range():
    for etype, weight in ENTITY_WEIGHTS.items():
        assert 0 < weight <= 1.0, f"{etype} weight {weight} out of (0, 1]"


def test_hard_block_types_subset_of_all():
    assert HARD_BLOCK_TYPES.issubset(ALL_ENTITY_TYPES)


def test_hard_block_types_contain_ssn_cc_account():
    assert SSN in HARD_BLOCK_TYPES
    assert CREDIT_CARD in HARD_BLOCK_TYPES
    assert ACCOUNT in HARD_BLOCK_TYPES


def test_hard_block_types_have_high_weights():
    for etype in HARD_BLOCK_TYPES:
        assert ENTITY_WEIGHTS[etype] >= 0.90, f"{etype} hard-block weight too low"


def test_risk_band_low():
    assert get_risk_band(0.0)  == "LOW"
    assert get_risk_band(0.5)  == "LOW"
    assert get_risk_band(0.99) == "LOW"


def test_risk_band_moderate():
    assert get_risk_band(1.0) == "MODERATE"
    assert get_risk_band(1.5) == "MODERATE"
    assert get_risk_band(1.99) == "MODERATE"


def test_risk_band_high():
    assert get_risk_band(2.0) == "HIGH"
    assert get_risk_band(2.5) == "HIGH"


def test_risk_band_critical():
    assert get_risk_band(3.0)  == "CRITICAL"
    assert get_risk_band(10.0) == "CRITICAL"


def test_risk_color_returns_ansi_string():
    for score in [0.5, 1.5, 2.5, 3.5]:
        color = get_risk_color(score)
        assert isinstance(color, str)
        assert color.startswith("\033[")

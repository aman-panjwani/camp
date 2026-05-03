from camp.core.registry import PIIRegistry, TurnRecord
from camp.core.extractor import DetectedEntity
from camp.core.entities import PERSON, LOCATION, SSN


def _entity(etype: str, value: str, turn: int = 0) -> DetectedEntity:
    return DetectedEntity(entity_type=etype, value=value, score=0.9, turn_index=turn)


def test_empty_registry():
    reg = PIIRegistry("test")
    assert len(reg) == 0
    assert reg.unique_types() == set()
    assert reg.all_entities() == []


def test_add_single_turn():
    reg = PIIRegistry("test")
    reg.add_turn(0, "John Smith", "[PERSON]", [_entity(PERSON, "John Smith")])
    assert len(reg) == 1


def test_add_turn_returns_record():
    reg    = PIIRegistry("test")
    record = reg.add_turn(0, "text", "text", [])
    assert isinstance(record, TurnRecord)
    assert record.turn_index == 0


def test_pii_types_at_turn_excludes_future():
    reg = PIIRegistry("test")
    reg.add_turn(0, "John Smith", "[PERSON]", [_entity(PERSON, "John Smith", 0)])
    reg.add_turn(1, "In Austin",  "[LOCATION]", [_entity(LOCATION, "Austin", 1)])

    types_t0 = reg.pii_types_at_turn(0)
    assert PERSON   in types_t0
    assert LOCATION not in types_t0


def test_pii_types_at_turn_includes_prior():
    reg = PIIRegistry("test")
    reg.add_turn(0, "John Smith", "[PERSON]", [_entity(PERSON, "John Smith", 0)])
    reg.add_turn(1, "In Austin",  "[LOCATION]", [_entity(LOCATION, "Austin", 1)])

    types_t1 = reg.pii_types_at_turn(1)
    assert PERSON   in types_t1
    assert LOCATION in types_t1


def test_unique_types():
    reg = PIIRegistry("test")
    reg.add_turn(0, "text", "text", [
        _entity(PERSON, "Alice"),
        _entity(SSN, "123-45-6789"),
    ])
    assert reg.unique_types() == {PERSON, SSN}


def test_cumulative_types_per_turn():
    reg = PIIRegistry("test")
    reg.add_turn(0, "t", "t", [_entity(PERSON, "Alice")])
    reg.add_turn(1, "t", "t", [_entity(LOCATION, "NYC")])

    cumulative = reg.cumulative_types_per_turn()
    assert PERSON   in cumulative[0]
    assert LOCATION not in cumulative[0]
    assert PERSON   in cumulative[1]
    assert LOCATION in cumulative[1]


def test_get_turn_by_index():
    reg = PIIRegistry("test")
    reg.add_turn(0, "a", "a", [])
    reg.add_turn(1, "b", "b", [])
    assert reg.get_turn(1).turn_index == 1
    assert reg.get_turn(99) is None


def test_all_turns_order():
    reg = PIIRegistry("test")
    reg.add_turn(0, "a", "a", [])
    reg.add_turn(1, "b", "b", [])
    turns = reg.all_turns()
    assert len(turns) == 2
    assert turns[0].turn_index == 0
    assert turns[1].turn_index == 1


def test_reset_clears_all():
    reg = PIIRegistry("test")
    reg.add_turn(0, "a", "a", [_entity(PERSON, "Alice")])
    reg.reset()
    assert len(reg) == 0
    assert reg.unique_types() == set()
    assert reg.all_entities() == []


def test_summary_keys():
    reg = PIIRegistry("session-42")
    reg.add_turn(0, "hello", "hello", [_entity(PERSON, "Bob")])
    s = reg.summary()
    assert s["session_id"] == "session-42"
    assert s["total_turns"] == 1
    assert s["turns_with_pii"] == 1

from camp.core.entities import EMAIL, LOCATION, PERSON
from camp.core.graph import PIICooccurrenceGraph


def test_empty_graph():
    g = PIICooccurrenceGraph()
    assert g.node_count() == 0
    assert g.edge_count() == 0


def test_single_entity_no_edges():
    g = PIICooccurrenceGraph(alpha=0.3)
    g.update({PERSON})
    assert g.node_count() == 1
    assert g.edge_count() == 0
    assert g.combination_amplifier(PERSON) == 1.0


def test_two_entities_create_one_edge():
    g = PIICooccurrenceGraph(alpha=0.3)
    g.update({PERSON, LOCATION})
    assert g.node_count() == 2
    assert g.edge_count() == 1


def test_three_entities_create_three_edges():
    g = PIICooccurrenceGraph(alpha=0.3)
    g.update({PERSON, LOCATION, EMAIL})
    assert g.node_count() == 3
    assert g.edge_count() == 3


def test_amplifier_formula():
    g = PIICooccurrenceGraph(alpha=0.3)
    g.update({PERSON, LOCATION, EMAIL})
    # PERSON is connected to LOCATION and EMAIL → degree=2
    # amplifier = 1 + 0.3 * 2 = 1.6
    assert abs(g.combination_amplifier(PERSON) - 1.6) < 0.001


def test_amplifier_single_node_is_one():
    g = PIICooccurrenceGraph(alpha=0.3)
    g.update({PERSON})
    assert g.combination_amplifier(PERSON) == 1.0


def test_graph_grows_monotonically():
    g = PIICooccurrenceGraph()
    g.update({PERSON})
    n1 = g.node_count()
    g.update({PERSON, LOCATION})
    n2 = g.node_count()
    g.update({PERSON, LOCATION, EMAIL})
    n3 = g.node_count()
    assert n1 <= n2 <= n3


def test_no_duplicate_edges():
    g = PIICooccurrenceGraph()
    g.update({PERSON, LOCATION})
    g.update({PERSON, LOCATION})
    assert g.edge_count() == 1


def test_unknown_entity_amplifier_returns_one():
    g = PIICooccurrenceGraph()
    assert g.combination_amplifier("TOTALLY_UNKNOWN_TYPE") == 1.0


def test_unknown_entity_degree_returns_zero():
    g = PIICooccurrenceGraph()
    assert g.degree("TOTALLY_UNKNOWN_TYPE") == 0


def test_snapshots_recorded_per_update():
    g = PIICooccurrenceGraph()
    g.update({PERSON})
    g.update({PERSON, LOCATION})
    assert len(g.snapshots()) == 2


def test_summary_keys():
    g = PIICooccurrenceGraph(alpha=0.5)
    g.update({PERSON, LOCATION})
    s = g.summary()
    assert "nodes" in s
    assert "edges" in s
    assert "alpha" in s
    assert "amplifiers" in s

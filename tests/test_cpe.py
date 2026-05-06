from camp.core.cpe import CPEScorer
from camp.core.entities import ENTITY_WEIGHTS, LOCATION, PERSON, SALARY
from camp.core.graph import PIICooccurrenceGraph


def _run(entity_sets: list, threshold: float = 2.0, alpha: float = 0.3):
    graph  = PIICooccurrenceGraph(alpha=alpha)
    scorer = CPEScorer(threshold=threshold)
    for i, types in enumerate(entity_sets):
        graph.update(types)
        scorer.update(graph, turn_index=i)
    return scorer, graph


def test_zero_score_empty_graph():
    graph  = PIICooccurrenceGraph()
    scorer = CPEScorer()
    assert scorer.compute(graph) == 0.0


def test_score_single_entity_no_edges():
    graph = PIICooccurrenceGraph(alpha=0.3)
    graph.update({PERSON})
    scorer = CPEScorer()
    # PERSON weight=0.60, degree=0, amplifier=1.0 → score=0.60
    expected = ENTITY_WEIGHTS[PERSON] * 1.0
    assert abs(scorer.compute(graph) - expected) < 0.001


def test_threshold_triggers():
    scorer, _ = _run([{PERSON}, {PERSON, LOCATION}, {PERSON, LOCATION, SALARY}], threshold=0.5)
    assert scorer.triggered()
    assert scorer.trigger_turn() is not None


def test_trigger_turn_recorded_only_once():
    scorer, graph = _run([{PERSON}, {PERSON, LOCATION}], threshold=0.5)
    first = scorer.trigger_turn()
    graph.update({PERSON, LOCATION, SALARY})
    scorer.update(graph, turn_index=99)
    assert scorer.trigger_turn() == first


def test_not_triggered_below_threshold():
    scorer, _ = _run([{PERSON}], threshold=5.0)
    assert not scorer.triggered()
    assert scorer.trigger_turn() is None


def test_history_length_matches_turns():
    scorer, _ = _run([{PERSON}, {LOCATION}, {SALARY}], threshold=10.0)
    assert len(scorer.history()) == 3


def test_current_score_matches_last_history():
    scorer, _ = _run([{PERSON}, {LOCATION}], threshold=10.0)
    assert scorer.current_score() == scorer.history()[-1]


def test_current_score_zero_when_empty():
    scorer = CPEScorer()
    assert scorer.current_score() == 0.0


def test_breakdown_has_correct_keys():
    graph = PIICooccurrenceGraph(alpha=0.3)
    graph.update({PERSON, LOCATION})
    scorer    = CPEScorer()
    breakdown = scorer.breakdown(graph)
    assert PERSON in breakdown
    for key in ("label", "weight", "degree", "amplifier", "contribution"):
        assert key in breakdown[PERSON]


def test_contribution_equals_weight_times_amplifier():
    graph = PIICooccurrenceGraph(alpha=0.3)
    graph.update({PERSON})
    scorer    = CPEScorer()
    breakdown = scorer.breakdown(graph)
    entry = breakdown[PERSON]
    expected = round(entry["weight"] * entry["amplifier"], 4)
    assert abs(entry["contribution"] - expected) < 0.001

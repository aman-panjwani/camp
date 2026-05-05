# camp.core.cpe
# -------------
# Cumulative PII Exposure scorer.
# Implements equation (2) from the CAMP paper:
#
#   CPE(t) = sum_v [ w(v) * (1 + alpha * degree(v)) ]
#
# Computes CPE after each turn using the current co-occurrence graph.

from typing import Dict, List, Optional

from camp.core.entities import ENTITY_WEIGHTS, ENTITY_LABELS, get_risk_band, get_risk_color, RESET
from camp.core.graph import PIICooccurrenceGraph


class CPEScorer:
    """
    Computes CPE score after each turn and tracks full score history.
    """

    def __init__(self, threshold: float = 2.0, weights: dict[str, float] | None = None) -> None:
        self.threshold = threshold
        self._weights  = {**ENTITY_WEIGHTS, **(weights or {})}
        self._history: List[float] = []
        self._triggered = False
        self._trigger_turn: Optional[int] = None

    def compute(self, graph: PIICooccurrenceGraph) -> float:
        """
        CPE(t) = sum_v [ w(v) * (1 + alpha * deg(v)) ]
        """
        score = 0.0
        for node in graph.nodes():
            weight    = self._weights.get(node, 0.3)
            amplifier = graph.combination_amplifier(node)
            score    += weight * amplifier
        return round(score, 4)

    def update(self, graph: PIICooccurrenceGraph, turn_index: int) -> float:
        """Compute CPE for this turn, store in history, check threshold."""
        score = self.compute(graph)
        self._history.append(score)

        if not self._triggered and score >= self.threshold:
            self._triggered    = True
            self._trigger_turn = turn_index

        return score

    def history(self) -> List[float]:
        return self._history

    def current_score(self) -> float:
        return self._history[-1] if self._history else 0.0

    def triggered(self) -> bool:
        return self._triggered

    def trigger_turn(self) -> Optional[int]:
        return self._trigger_turn

    def breakdown(self, graph: PIICooccurrenceGraph) -> Dict[str, dict]:
        """Per-entity contribution breakdown for inspection and paper tables."""
        result = {}
        for node in graph.nodes():
            weight    = self._weights.get(node, 0.3)
            amplifier = graph.combination_amplifier(node)
            result[node] = {
                "label":        ENTITY_LABELS.get(node, node),
                "weight":       weight,
                "degree":       graph.degree(node),
                "amplifier":    round(amplifier, 3),
                "contribution": round(weight * amplifier, 4),
            }
        return result

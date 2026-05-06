# camp.core.graph
# ---------------
# PII Co-occurrence Graph.
# Nodes = PII entity types seen in session.
# Edges = two entity types that co-exist in the same session.
# Graph grows monotonically across turns.
# Used to compute combination amplifier for CPE score:
#   f(v) = 1 + alpha * degree(v)


import networkx as nx

from camp.core.entities import ENTITY_LABELS, ENTITY_WEIGHTS


class PIICooccurrenceGraph:
    """
    Maintains a co-occurrence graph over PII entity types
    detected across a conversation session.

    Nodes  : PII entity types (strings)
    Edges  : pair of entity types that have both appeared in the session
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.graph: nx.Graph = nx.Graph()
        self._snapshots: list[nx.Graph] = []

    def update(self, entity_types: set[str]) -> None:
        """
        Update graph with entity types accumulated at current turn.
        Adds new nodes and edges between all pairs now in the session.
        """
        for etype in entity_types:
            if not self.graph.has_node(etype):
                self.graph.add_node(
                    etype,
                    weight=ENTITY_WEIGHTS.get(etype, 0.3),
                    label=ENTITY_LABELS.get(etype, etype),
                )

        types_list = list(entity_types)
        for i in range(len(types_list)):
            for j in range(i + 1, len(types_list)):
                if not self.graph.has_edge(types_list[i], types_list[j]):
                    self.graph.add_edge(types_list[i], types_list[j])

        self._snapshots.append(self.graph.copy())

    def combination_amplifier(self, entity_type: str) -> float:
        """f(v) = 1 + alpha * degree(v)"""
        if not self.graph.has_node(entity_type):
            return 1.0
        return 1.0 + self.alpha * self.graph.degree(entity_type)

    def nodes(self) -> list[str]:
        return list(self.graph.nodes())

    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges())

    def degree(self, entity_type: str) -> int:
        if not self.graph.has_node(entity_type):
            return 0
        return self.graph.degree(entity_type)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def snapshots(self) -> list[nx.Graph]:
        return self._snapshots

    def summary(self) -> dict:
        return {
            "nodes":      self.nodes(),
            "edges":      [list(e) for e in self.edges()],
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "alpha":      self.alpha,
            "amplifiers": {
                node: round(self.combination_amplifier(node), 3)
                for node in self.nodes()
            },
        }

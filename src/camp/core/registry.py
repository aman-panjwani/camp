# camp.core.registry
# ------------------
# Session-level PII registry.
# Tracks every PII entity detected across all turns.
# Never transmitted to external models.
# Single source of truth for graph, CPE scorer, and pseudonymizer.

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from camp.core.extractor import DetectedEntity


@dataclass
class TurnRecord:
    turn_index:  int
    raw_text:    str
    masked_text: str
    entities:    list[DetectedEntity] = field(default_factory=list)

    @property
    def entity_types(self) -> set[str]:
        return {e.entity_type for e in self.entities}

    @property
    def has_pii(self) -> bool:
        return len(self.entities) > 0


class PIIRegistry:
    """
    Stateful record of all PII detected across every turn
    in a single conversation session.
    """

    def __init__(self, session_id: str = "session") -> None:
        self.session_id = session_id
        self._turns: list[TurnRecord] = []
        self._type_index: dict[str, list[DetectedEntity]] = defaultdict(list)

    def add_turn(
        self,
        turn_index:  int,
        raw_text:    str,
        masked_text: str,
        entities:    list[DetectedEntity],
    ) -> TurnRecord:
        record = TurnRecord(
            turn_index=turn_index,
            raw_text=raw_text,
            masked_text=masked_text,
            entities=entities,
        )
        self._turns.append(record)

        for entity in entities:
            self._type_index[entity.entity_type].append(entity)

        return record

    def all_entities(self) -> list[DetectedEntity]:
        result = []
        for record in self._turns:
            result.extend(record.entities)
        return result

    def unique_types(self) -> set[str]:
        return set(self._type_index.keys())

    def entities_by_type(self) -> dict[str, list[DetectedEntity]]:
        return dict(self._type_index)

    def get_turn(self, turn_index: int) -> TurnRecord | None:
        for record in self._turns:
            if record.turn_index == turn_index:
                return record
        return None

    def all_turns(self) -> list[TurnRecord]:
        return self._turns

    def pii_types_at_turn(self, turn_index: int) -> set[str]:
        """All unique PII types accumulated from turn 0 through turn_index."""
        types: set[str] = set()
        for record in self._turns:
            if record.turn_index <= turn_index:
                types.update(record.entity_types)
        return types

    def cumulative_types_per_turn(self) -> list[frozenset[str]]:
        """List where index i contains all unique PII types seen through turn i."""
        cumulative = []
        seen: set[str] = set()
        for record in self._turns:
            seen = seen | record.entity_types
            cumulative.append(frozenset(seen))
        return cumulative

    def summary(self) -> dict[str, Any]:
        return {
            "session_id":        self.session_id,
            "total_turns":       len(self._turns),
            "turns_with_pii":    sum(1 for t in self._turns if t.has_pii),
            "total_entities":    len(self.all_entities()),
            "unique_types":      sorted(self.unique_types()),
            "entities_per_type": {k: len(v) for k, v in self._type_index.items()},
        }

    def reset(self) -> None:
        self._turns.clear()
        self._type_index.clear()

    def __len__(self) -> int:
        return len(self._turns)

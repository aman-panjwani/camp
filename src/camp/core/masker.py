# camp.core.masker
# ----------------
# CAMP masking decision engine - central coordinator.
#
# Decision logic per turn:
#   PASS         CPE below threshold - send original text
#   PSEUDONYMIZE CPE crossed threshold - rewrite full history with pseudonyms
#   BLOCK        Hard-block entity detected - always block regardless of CPE

from dataclasses import dataclass
from typing import List, Optional

from camp.core.extractor import extract_pii, mask_text, DetectedEntity
from camp.core.registry import PIIRegistry, TurnRecord
from camp.core.graph import PIICooccurrenceGraph
from camp.core.cpe import CPEScorer
from camp.core.pseudonymizer import Pseudonymizer
from camp.core.entities import get_risk_band, HARD_BLOCK_TYPES

PASS         = "PASS"
PSEUDONYMIZE = "PSEUDONYMIZE"
BLOCK        = "BLOCK"


@dataclass
class TurnResult:
    """Complete record of how a single conversation turn was handled."""
    turn_index:        int
    raw_text:          str
    sent_to_llm:       str
    entities:          List[DetectedEntity]
    cpe_score:         float
    risk_band:         str
    decision:          str
    triggered:         bool
    trigger_turn:      Optional[int]
    rewritten_history: Optional[List[str]]


class CAMPMasker:
    """
    Central CAMP coordinator.

    Processes each conversation turn through the full pipeline:
      1. Extract PII locally (never leaves device)
      2. Update registry and co-occurrence graph
      3. Compute CPE score
      4. Decide: PASS / PSEUDONYMIZE / BLOCK
      5. Return what gets sent to the LLM

    Usage:
        masker = CAMPMasker(threshold=2.0, alpha=0.3)
        result = masker.process_turn("My name is Sarah Johnson", turn_index=0)
        llm_input = result.sent_to_llm  # safe to send
        real_response = masker.demask_response(llm_response)
    """

    def __init__(
        self,
        threshold:   float            = 2.0,
        alpha:       float            = 0.3,
        session_id:  str              = "session",
        redaction_map: dict[str, str] | None = None,
        extra_patterns: list[dict] | None = None,
    ) -> None:
        self.threshold       = threshold
        self._redaction_map  = redaction_map
        self._extra_patterns = extra_patterns  # caller-defined regex patterns
        self.registry      = PIIRegistry(session_id=session_id)
        self.graph         = PIICooccurrenceGraph(alpha=alpha)
        self.scorer        = CPEScorer(threshold=threshold)
        self.pseudonymizer = Pseudonymizer(redaction_map=redaction_map)
        self._results: List[TurnResult] = []

    def process_turn(self, text: str, turn_index: int) -> TurnResult:
        """
        Process a single conversation turn through the CAMP pipeline.

        Args:
            text:       Raw user message.
            turn_index: Turn number in the conversation (0-based).

        Returns:
            TurnResult with the full decision and what to send to the LLM.
        """
        # Step 1 - Extract PII locally
        entities = extract_pii(text, turn_index, self._extra_patterns)

        # Step 2 - Update registry
        self.registry.add_turn(
            turn_index=turn_index,
            raw_text=text,
            masked_text=mask_text(text, entities, self._redaction_map),
            entities=entities,
        )

        # Step 3 - Update graph and compute CPE
        accumulated = self.registry.pii_types_at_turn(turn_index)
        self.graph.update(accumulated)
        cpe_score = self.scorer.update(self.graph, turn_index)
        risk_band = get_risk_band(cpe_score)

        # Step 4 - Decide masking action
        has_hard_block    = any(e.is_hard_block(self._redaction_map) for e in entities)
        rewritten_history = None

        if self.scorer.triggered():
            decision          = PSEUDONYMIZE
            rewritten_history = self.pseudonymizer.rewrite_history(
                self.registry.all_turns()
            )
            sent_to_llm = rewritten_history[turn_index]
        elif has_hard_block:
            decision    = BLOCK
            sent_to_llm = mask_text(text, entities, self._redaction_map)
        else:
            decision    = PASS
            sent_to_llm = text

        result = TurnResult(
            turn_index=turn_index,
            raw_text=text,
            sent_to_llm=sent_to_llm,
            entities=entities,
            cpe_score=cpe_score,
            risk_band=risk_band,
            decision=decision,
            triggered=self.scorer.triggered(),
            trigger_turn=self.scorer.trigger_turn(),
            rewritten_history=rewritten_history,
        )

        self._results.append(result)
        return result

    def demask_response(self, response: str) -> str:
        """Restore real identities in an LLM response."""
        return self.pseudonymizer.demask_response(response)

    def results(self) -> List[TurnResult]:
        return self._results

    def cpe_history(self) -> List[float]:
        return self.scorer.history()

    def trigger_turn(self) -> Optional[int]:
        return self.scorer.trigger_turn()

    def pseudonym_map(self) -> dict:
        return self.pseudonymizer.pseudonym_map()

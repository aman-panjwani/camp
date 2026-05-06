# camp.core.masker
# ----------------
# CAMP masking decision engine - central coordinator.
#
# Decision logic per turn:
#   PASS         CPE below threshold - send original text
#   PSEUDONYMIZE CPE crossed threshold - rewrite full history with pseudonyms
#   BLOCK        Hard-block entity detected - always block regardless of CPE

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

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
        threshold:      float                 = 2.0,
        alpha:          float                 = 0.3,
        session_id:     str                   = "session",
        redaction_map:  dict[str, str] | None = None,
        custom_patterns: list[dict]    | None  = None,
        entity_weights: dict[str, float] | None = None,
    ) -> None:
        self.threshold       = threshold
        self._alpha          = alpha
        self._session_id     = session_id
        self._redaction_map  = redaction_map
        self._custom_patterns = custom_patterns
        self._entity_weights = entity_weights
        self.registry      = PIIRegistry(session_id=session_id)
        self.graph         = PIICooccurrenceGraph(alpha=alpha)
        self.scorer        = CPEScorer(threshold=threshold, weights=entity_weights)
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
        entities = extract_pii(text, turn_index, self._custom_patterns)

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

    def reset(self) -> None:
        """Clear all session state. Config (threshold, alpha, etc.) is preserved."""
        self.registry      = PIIRegistry(session_id=self._session_id)
        self.graph         = PIICooccurrenceGraph(alpha=self._alpha)
        self.scorer        = CPEScorer(threshold=self.threshold, weights=self._entity_weights)
        self.pseudonymizer = Pseudonymizer(redaction_map=self._redaction_map)
        self._results      = []

    # ── Tool-call helpers ─────────────────────────────────────────────────────

    def demask_args(self, args: dict) -> dict:
        """Recursively restore real values in tool call arguments (handles nested dicts/lists)."""
        def _walk(obj: Any) -> Any:
            if isinstance(obj, str):
                return self.pseudonymizer.demask_response(obj)
            if isinstance(obj, dict):
                return {k: _walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_walk(item) for item in obj]
            return obj
        return _walk(args)

    def mask_content(self, content: str | dict | list) -> str:
        """Replace real PII in a tool result with session pseudonyms before sending back to LLM."""
        text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        pmap = self.pseudonymizer.pseudonym_map()
        for real, fake in sorted(pmap.items(), key=lambda x: len(x[0]), reverse=True):
            if real in text:
                text = text.replace(real, fake)
        return text

    def build_tool_result(self, tool_use_id: str, content: str | dict | list) -> dict:
        """Format a masked tool result as an Anthropic tool_result message."""
        return {
            "type":        "tool_result",
            "tool_use_id": tool_use_id,
            "content":     content if isinstance(content, str) else json.dumps(content),
        }

    def process_tool_call(self, tool_use_id: str, args: dict, fn: Callable) -> dict:
        """Sync: demask args -> call fn(**args) -> mask result -> format tool_result."""
        real_args = self.demask_args(args)
        result    = fn(**real_args)
        return self.build_tool_result(tool_use_id, self.mask_content(result))

    async def process_tool_call_async(self, tool_use_id: str, args: dict, fn: Any) -> dict:
        """Async: demask args -> await fn(**args) -> mask result -> format tool_result."""
        real_args = self.demask_args(args)
        result    = await fn(**real_args)
        return self.build_tool_result(tool_use_id, self.mask_content(result))

"""
camp.integrations.llm
---------------------
Framework-agnostic CAMP wrapper for any LLM callable.
No optional dependencies required - works with any LLM.

Pattern 1 - process only (you own the LLM call):
    session = CAMPSession(threshold=2.0)
    result  = session.process("My name is Sarah Johnson")
    raw     = your_llm(result.sent_to_llm)
    clean   = session.demask(raw)

Pattern 2 - bound callable:
    session  = CAMPSession.wrap(your_llm_fn, threshold=2.0)
    response = session.chat("My SSN is 512-34-7891")
"""

from __future__ import annotations

from collections.abc import Callable

from camp.core.masker import CAMPMasker, TurnResult


class CAMPSession:
    """
    Wraps any LLM callable with session-aware CAMP PII protection.

    Tracks cumulative PII exposure across the full conversation.
    Masks inputs before they reach the LLM and demaskes responses
    before returning them to the user.
    """

    def __init__(
        self,
        threshold:   float                       = 2.0,
        alpha:       float                       = 0.3,
        session_id:  str                         = "default",
        llm_fn:      Callable[..., str] | None = None,
        redaction_map: dict[str, str] | None             = None,
    ) -> None:
        self._threshold   = threshold
        self._alpha       = alpha
        self._session_id  = session_id
        self._llm_fn      = llm_fn
        self._redaction_map = redaction_map
        self._masker      = CAMPMasker(
            threshold=threshold, alpha=alpha,
            session_id=session_id, redaction_map=redaction_map,
        )
        self._turn_index = 0

    @classmethod
    def wrap(
        cls,
        llm_fn:      Callable[..., str],
        threshold:   float           = 2.0,
        alpha:       float           = 0.3,
        session_id:  str             = "default",
        redaction_map: dict[str, str] | None = None,
    ) -> CAMPSession:
        """Create a CAMPSession bound to a specific LLM callable."""
        return cls(
            threshold=threshold, alpha=alpha,
            session_id=session_id, llm_fn=llm_fn,
            redaction_map=redaction_map,
        )

    def process(self, message: str) -> TurnResult:
        """
        Apply CAMP to a message and return the TurnResult.
        Does NOT call the LLM - use when you manage the LLM call yourself.
        """
        result = self._masker.process_turn(message, self._turn_index)
        self._turn_index += 1
        return result

    def demask(self, response: str) -> str:
        """Restore real identities in an LLM response."""
        return self._masker.demask_response(response)

    def chat(self, message: str, **kwargs: object) -> str:
        """
        Process message through CAMP, call the bound LLM, demask the response.

        Raises:
            RuntimeError: if no llm_fn is bound.
        """
        if self._llm_fn is None:
            raise RuntimeError(
                "No LLM callable bound. Use CAMPSession.wrap(llm_fn) "
                "or pass llm_fn= to the constructor."
            )
        result      = self.process(message)
        raw_response = self._llm_fn(result.sent_to_llm, **kwargs)
        return self.demask(raw_response)

    def reset(self) -> None:
        """Start a new session with the same configuration."""
        self._masker = CAMPMasker(
            threshold=self._threshold,
            alpha=self._alpha,
            session_id=self._session_id,
            redaction_map=self._redaction_map,
        )
        self._turn_index = 0

    @property
    def cpe_score(self) -> float:
        history = self._masker.cpe_history()
        return history[-1] if history else 0.0

    @property
    def triggered(self) -> bool:
        return self._masker.scorer.triggered()

    @property
    def turn_count(self) -> int:
        return self._turn_index

    @property
    def masker(self) -> CAMPMasker:
        return self._masker

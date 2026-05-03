"""
camp
----
CAMP: Cumulative Agentic Masking and Pruning

Session-aware PII protection for LLM pipelines.
Intercepts, scores, and pseudonymizes PII before it reaches any LLM -
then restores real identities in responses.

Quick start:
    from camp import CAMPMasker, CAMPSession

    # Standalone
    masker = CAMPMasker(threshold=2.0)
    result = masker.process_turn("My name is Sarah Johnson", turn_index=0)
    print(result.sent_to_llm)   # pseudonymized or original

    # Wrap any LLM callable
    session = CAMPSession.wrap(my_llm_fn, threshold=2.0)
    response = session.chat("My SSN is 512-34-7891")
"""

from camp.core.masker import CAMPMasker, TurnResult
from camp.integrations.llm import CAMPSession
from camp.core.entities import DEFAULT_REDACTION_MAP, HARD_BLOCK_TYPES

__all__ = ["CAMPMasker", "TurnResult", "CAMPSession", "DEFAULT_REDACTION_MAP", "HARD_BLOCK_TYPES"]
__version__ = "0.1.1"
__author__ = "Aman Panjwani"

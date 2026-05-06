"""
camp.integrations.langchain
---------------------------
LangChain integration for CAMP PII protection.

Requires: pip install campii[langchain]

Two integration styles:

1. CAMPCallbackHandler - attach to any existing chain/LLM:
    handler = CAMPCallbackHandler(threshold=2.0)
    chain   = ConversationChain(llm=llm, callbacks=[handler])
    result  = chain.invoke({"input": "My SSN is 512-34-7891"})

2. CAMPChain - wrap any LangChain runnable:
    protected = CAMPChain.from_runnable(chain, threshold=2.0)
    result    = protected.invoke({"input": "My name is Sarah Johnson"})
    print(protected.handler.cpe_score)
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from camp.core.masker import CAMPMasker, TurnResult

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install campii[langchain]"
    ) from exc


class CAMPCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that applies CAMP PII protection.

    Intercepts prompts in on_llm_start / on_chat_model_start (mutates in-place),
    and demaskes LLM outputs in on_llm_end.

    Compatible with any LangChain chain, agent, or LLM object.
    """

    def __init__(
        self,
        threshold:   float            = 2.0,
        alpha:       float            = 0.3,
        session_id:  str              = "default",
        redaction_map: dict[str, str] | None  = None,
        masker:      CAMPMasker | None = None,
    ) -> None:
        super().__init__()
        self._masker = masker or CAMPMasker(
            threshold=threshold, alpha=alpha,
            session_id=session_id, redaction_map=redaction_map,
        )
        self._turn_index          = 0
        self._last_result:        TurnResult | None = None
        self._last_raw_llm_output: str = ""

    # ── LLM (non-chat) ────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        for i, prompt in enumerate(prompts):
            result            = self._masker.process_turn(prompt, self._turn_index)
            self._last_result = result
            prompts[i]        = result.sent_to_llm
        self._turn_index += 1

    # ── Chat model ────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        for message_list in messages:
            if message_list:
                last = message_list[-1]
                if hasattr(last, "content") and isinstance(last.content, str):
                    result            = self._masker.process_turn(last.content, self._turn_index)
                    self._last_result = result
                    last.content      = result.sent_to_llm
        self._turn_index += 1

    # ── Response ──────────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._last_raw_llm_output = ""
        for generation_list in response.generations:
            for generation in generation_list:
                if hasattr(generation, "text"):
                    if not self._last_raw_llm_output:
                        self._last_raw_llm_output = generation.text
                    generation.text = self._masker.demask_response(generation.text)
                if hasattr(generation, "message"):
                    msg = generation.message
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        if not self._last_raw_llm_output:
                            self._last_raw_llm_output = msg.content
                        msg.content = self._masker.demask_response(msg.content)

    # ── Inspection ────────────────────────────────────────────

    @property
    def cpe_score(self) -> float:
        history = self._masker.cpe_history()
        return history[-1] if history else 0.0

    @property
    def triggered(self) -> bool:
        return self._masker.scorer.triggered()

    @property
    def last_result(self) -> TurnResult | None:
        return self._last_result

    @property
    def last_raw_llm_output(self) -> str:
        return self._last_raw_llm_output

    @property
    def masker(self) -> CAMPMasker:
        return self._masker


class CAMPChain:
    """
    High-level wrapper: adds CAMP protection to any LangChain runnable.

    Usage:
        from camp.integrations.langchain import CAMPChain

        protected = CAMPChain.from_runnable(my_chain, threshold=2.0)
        result    = protected.invoke({"input": "My SSN is 512-34-7891"})
        print(protected.handler.cpe_score)
    """

    def __init__(self, runnable: Any, handler: CAMPCallbackHandler) -> None:
        self._runnable = runnable
        self._handler  = handler

    @classmethod
    def from_runnable(
        cls,
        runnable:    Any,
        threshold:   float           = 2.0,
        alpha:       float           = 0.3,
        session_id:  str             = "default",
        redaction_map: dict[str, str] | None = None,
    ) -> CAMPChain:
        handler = CAMPCallbackHandler(
            threshold=threshold, alpha=alpha,
            session_id=session_id, redaction_map=redaction_map,
        )
        return cls(runnable, handler)

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks.append(self._handler)
        return self._runnable.invoke(inputs, callbacks=callbacks, **kwargs)

    async def ainvoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks.append(self._handler)
        return await self._runnable.ainvoke(inputs, callbacks=callbacks, **kwargs)

    @property
    def handler(self) -> CAMPCallbackHandler:
        return self._handler

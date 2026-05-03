# CAMP

<p align="center">
  <strong>Cumulative Agentic Masking and Pruning</strong><br>
  Session-aware PII protection for LLM pipelines
</p>

<p align="center">
  <a href="https://pypi.org/project/camp/"><img src="https://img.shields.io/pypi/v/camp?label=pypi&color=blue" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://pypi.org/project/camp/"><img src="https://img.shields.io/badge/python-3.11%20|%203.12-blue" alt="Python"></a>
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-2026-b31b1b" alt="arXiv"></a>
</p>

---

CAMP tracks cumulative PII exposure across an entire conversation - not just a single message - and pseudonymizes the full history the moment risk crosses a configurable threshold. Real identities never leave your machine.

---

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Integrations](#integrations)
  - [Any LLM callable](#integration-1---any-llm-callable)
  - [LangChain](#integration-2---langchain)
  - [Microsoft Agent Framework](#integration-3---microsoft-agent-framework)
- [Configuration](#configuration)
- [Supported entity types](#supported-entity-types)
- [Development](#development)
- [Research](#research)
- [License](#license)

---

## How it works

Every conversation turn, CAMP runs a four-step pipeline entirely on-device:

1. **Extract** - detects PII locally using Microsoft Presidio and spaCy NER, plus custom regex recognizers for financial and corporate data
2. **Graph** - updates a co-occurrence graph where nodes are entity types and edges form when types appear together across turns
3. **Score** - computes a Cumulative PII Exposure (CPE) score using the formula below
4. **Decide** - takes one of three actions per turn

```
CPE(t) = Σ w(v) × (1 + α × degree(v))
```

| Decision | Condition | Action |
|---|---|---|
| `PASS` | CPE below threshold | Send original text to LLM |
| `PSEUDONYMIZE` | CPE crossed threshold | Rewrite full conversation history with consistent synthetic identities |
| `BLOCK` | Hard-block entity detected | Redact immediately, regardless of CPE score |

Hard-blocked types (always redacted): `US_SSN`, `CREDIT_CARD`, `ACCOUNT_NUMBER`

---

## Installation

**Requirements:** Python 3.11+

```bash
pip install camp
```

CAMP uses spaCy for named entity recognition. Download the required model after installation:

```bash
python -m spacy download en_core_web_lg
```

### Optional extras

| Extra | Command | Adds |
|---|---|---|
| LangChain | `pip install camp[langchain]` | `CAMPCallbackHandler`, `CAMPChain` |
| Agent Framework | `pip install camp[agent-framework]` | `CAMPAgentMiddleware` |
| All integrations | `pip install camp[all]` | Everything above |

---

## Quick start

```python
from camp import CAMPMasker

masker = CAMPMasker(threshold=2.0, alpha=0.3)

conversation = [
    "Hi, I need help with my bank account.",
    "My name is Michael Torres.",
    "I bank with Chase, account ending in 4872.",
    "I live in Austin, Texas.",
    "My SSN is 512-34-7891.",
]

for i, text in enumerate(conversation):
    result = masker.process_turn(text, turn_index=i)
    print(f"Turn {i}  [{result.decision:13}]  CPE={result.cpe_score:.2f}  |  {result.sent_to_llm}")

# Restore real identities in the LLM response before showing to the user
llm_response = "I can help you with that, Michael."
clean = masker.demask_response(llm_response)
```

**Example output:**

```
Turn 0  [PASS         ]  CPE=0.00  |  Hi, I need help with my bank account.
Turn 1  [PASS         ]  CPE=0.60  |  My name is Michael Torres.
Turn 2  [BLOCK        ]  CPE=1.55  |  I bank with Chase, account ending in [BLOCKED].
Turn 3  [PASS         ]  CPE=1.60  |  I live in Austin, Texas.
Turn 4  [BLOCK        ]  CPE=2.60  |  My SSN is [BLOCKED].
```

---

## Integrations

### Integration 1 - Any LLM callable

`CAMPSession` wraps any function that accepts a string and returns a string. No framework dependency required.

```python
from camp import CAMPSession
import openai

client = openai.OpenAI()

def my_llm(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Wrap once - protection is applied automatically on every call
session = CAMPSession.wrap(my_llm, threshold=2.0, alpha=0.3)

response = session.chat("My name is Sarah Johnson")
response = session.chat("I live in Denver, Colorado")
response = session.chat("My SSN is 512-34-7891")  # blocked, LLM is never called

print(f"CPE score : {session.cpe_score:.2f}")
print(f"Triggered : {session.triggered}")
```

**Manual mode** - manage the LLM call yourself:

```python
result = session.process("My email is sarah@example.com")
raw    = my_llm(result.sent_to_llm)   # call LLM with masked text
clean  = session.demask(raw)           # restore real identity in the response
```

---

### Integration 2 - LangChain

Requires `pip install camp[langchain]`

**Option A - callback handler** (attach to any existing chain or LLM):

```python
from camp.integrations.langchain import CAMPCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

handler = CAMPCallbackHandler(threshold=2.0)
chain   = ConversationChain(llm=ChatOpenAI(model="gpt-4o"), callbacks=[handler])

chain.invoke({"input": "My name is Sarah Johnson"})
chain.invoke({"input": "I live in Denver, Colorado"})
chain.invoke({"input": "My SSN is 512-34-7891"})

print(f"CPE           : {handler.cpe_score:.2f}")
print(f"Last decision : {handler.last_result.decision}")
```

**Option B - CAMPChain wrapper** (one-liner setup):

```python
from camp.integrations.langchain import CAMPChain

protected = CAMPChain.from_runnable(chain, threshold=2.0)
result    = protected.invoke({"input": "My SSN is 512-34-7891"})

print(protected.handler.triggered)
```

---

### Integration 3 - Microsoft Agent Framework

Requires `pip install camp[agent-framework]`

**Class-based middleware** (recommended - maintains session state across all runs):

```python
from camp.integrations.agent_framework import CAMPAgentMiddleware
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
import asyncio

async def main():
    async with (
        AzureCliCredential() as credential,
        Agent(
            client=FoundryChatClient(credential=credential),
            name="SupportAgent",
            instructions="You are a helpful customer support assistant.",
            middleware=[CAMPAgentMiddleware(threshold=2.0, alpha=0.3)],
        ) as agent,
    ):
        await agent.run("My name is Sarah Johnson")
        await agent.run("I live in Denver, Colorado")
        await agent.run("My SSN is 512-34-7891")
        # ^ Blocked before reaching the agent; returns a safe refusal message

        camp = agent.middleware[0]
        print(f"CPE score  : {camp.cpe_score:.2f}")
        print(f"Triggered  : {camp.triggered}")
        print(f"Pseudonyms : {camp.pseudonym_map}")

asyncio.run(main())
```

**Function-based factory** (lightweight, per-run):

```python
from camp.integrations.agent_framework import create_camp_middleware

camp   = create_camp_middleware(threshold=1.5)
result = await agent.run("My name is Sarah Johnson", middleware=[camp])
```

---

## Configuration

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `2.0` | CPE score at which pseudonymization triggers |
| `alpha` | `0.3` | Graph amplifier - controls how much entity co-occurrence raises the score |
| `session_id` | `"default"` | Session label used in the PII registry |
| `redaction_map` | `None` | Override default hard-block replacements |
| `extra_patterns` | `None` | Additional regex recognizers for domain-specific PII |

### Risk bands

| CPE range | Band |
|---|---|
| 0.0 - 1.0 | LOW |
| 1.0 - 2.0 | MODERATE |
| 2.0 - 3.0 | HIGH |
| 3.0+ | CRITICAL |

### Custom recognizers

Pass domain-specific patterns at construction time:

```python
masker = CAMPMasker(
    threshold=2.0,
    extra_patterns=[
        {"entity": "EMPLOYEE_ID", "pattern": r"\bEMP-\d{6}\b", "score": 0.9},
        {"entity": "PROJECT_CODE", "pattern": r"\bPRJ-[A-Z]{3}-\d{4}\b", "score": 0.85},
    ],
)
```

---

## Supported entity types

| Category | Entity types |
|---|---|
| **Identity** | Person name, Date of birth, SSN, Driver license, Ethnicity |
| **Contact** | Email address, Phone number, Location, IP address |
| **Financial** | Credit card, Account number, IBAN, SWIFT/BIC, Crypto wallet, Transaction ID, US ITIN |
| **Employment** | Salary, Age, Organization |
| **Medical** | Medical condition |
| **Corporate** | Financial amount, Financial metric, Internal projection, Confidential data |

---

## Development

```bash
git clone https://github.com/aman-panjwani/camp
cd camp
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

**Run the test suite:**

```bash
# Unit tests (no spaCy model required - Presidio is mocked)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=camp --cov-report=term-missing
```

**Lint and type-check:**

```bash
ruff check src/ tests/
mypy src/
```

---

## Research

CAMP is the reference implementation for the following paper:

```bibtex
@article{panjwani2026camp,
  title   = {CAMP: Cumulative Agentic Masking and Pruning for Session-Aware PII Protection in LLM Pipelines},
  author  = {Panjwani, Aman},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT

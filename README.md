# camp

**CAMP: Cumulative Agentic Masking and Pruning**

Session-aware PII protection for LLM pipelines. CAMP tracks cumulative PII exposure across an entire conversation - not just a single turn - and pseudonymizes the full history the moment the risk crosses a threshold. Real identities never leave your machine.

> Aman Panjwani · Independent Researcher · arXiv 2026

---

## How it works

Every turn, CAMP:
1. **Extracts PII locally** using Microsoft Presidio + spaCy
2. **Updates a co-occurrence graph** - nodes are entity types, edges form when types appear together
3. **Scores cumulative risk** `CPE(t) = Σ w(v) × (1 + α × degree(v))`
4. **Decides** per turn:
   - `PASS` - CPE below threshold -> send original text
   - `PSEUDONYMIZE` - CPE crossed threshold -> rewrite full history with consistent fake identities
   - `BLOCK` - hard-block type (SSN, credit card, account number) -> always redact, regardless of CPE

---

## Installation

```bash
pip install camp

# Required: download the spaCy model
python -m spacy download en_core_web_lg
```

**Optional extras:**

```bash
pip install camp[langchain]          # LangChain integration
pip install camp[agent-framework]    # Microsoft Agent Framework integration
pip install camp[all]                # All integrations
```

---

## Quick start - standalone

```python
from camp import CAMPMasker

masker = CAMPMasker(threshold=2.0, alpha=0.3)

conversation = [
    "Hi I need help with my bank account.",
    "My name is Michael Torres.",
    "I bank with Chase, account ending in 4872.",
    "I live in Austin, Texas.",
    "My SSN is 512-34-7891.",
]

for i, text in enumerate(conversation):
    result = masker.process_turn(text, turn_index=i)
    print(f"Turn {i} [{result.decision}] CPE={result.cpe_score:.2f}")
    print(f"  → Sent to LLM: {result.sent_to_llm}")

# Demask LLM response before showing to user
llm_response = "I can help you with that, Michael."
real_response = masker.demask_response(llm_response)
```

---

## Integration 1 - Any LLM callable (`CAMPSession`)

Works with OpenAI, Anthropic, Google, or any function that takes a string and returns a string.

```python
from camp import CAMPSession
import openai

client = openai.OpenAI()

def my_llm(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Wrap once - all calls auto-protected
session = CAMPSession.wrap(my_llm, threshold=2.0, alpha=0.3)

response = session.chat("My name is Sarah Johnson")
response = session.chat("I live in Denver, Colorado")
response = session.chat("My SSN is 512-34-7891")  # → [BLOCKED], LLM never called

print(f"CPE score: {session.cpe_score:.2f}")
print(f"Triggered: {session.triggered}")

# Or manage the LLM call yourself:
result = session.process("My email is sarah@example.com")
raw    = my_llm(result.sent_to_llm)          # send masked text
clean  = session.demask(raw)                  # restore real identity in response
```

---

## Integration 2 - LangChain

```python
from camp.integrations.langchain import CAMPCallbackHandler, CAMPChain
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4o")

# Option A - callback handler (attach to any existing chain)
handler = CAMPCallbackHandler(threshold=2.0)
chain   = ConversationChain(llm=llm, callbacks=[handler])

chain.invoke({"input": "My name is Sarah Johnson"})
chain.invoke({"input": "I live in Denver, Colorado"})
chain.invoke({"input": "My SSN is 512-34-7891"})

print(f"CPE: {handler.cpe_score:.2f}")
print(f"Last decision: {handler.last_result.decision}")

# Option B - CAMPChain wrapper (one-liner setup)
protected = CAMPChain.from_runnable(chain, threshold=2.0)
result    = protected.invoke({"input": "My SSN is 512-34-7891"})
print(protected.handler.triggered)
```

---

## Integration 3 - Microsoft Agent Framework

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
        # All runs automatically protected - real PII never reaches the LLM
        result = await agent.run("My name is Sarah Johnson")
        result = await agent.run("I live in Denver, Colorado")
        result = await agent.run("My SSN is 512-34-7891")
        # ↑ Blocked before reaching agent. Returns safe refusal message.

        camp = agent.middleware[0]  # type: CAMPAgentMiddleware
        print(f"CPE score : {camp.cpe_score:.2f}")
        print(f"Triggered : {camp.triggered}")
        print(f"Pseudonyms: {camp.pseudonym_map}")

asyncio.run(main())
```

**Per-run middleware** (single run only):

```python
from camp.integrations.agent_framework import create_camp_middleware

camp   = create_camp_middleware(threshold=1.5)
result = await agent.run("My name is Sarah Johnson", middleware=[camp])
```

---

## Configuration

| Parameter   | Default | Description |
|-------------|---------|-------------|
| `threshold` | `2.0`   | CPE score above which pseudonymization triggers |
| `alpha`     | `0.3`   | Graph connectivity amplifier - higher = more sensitive to entity combinations |
| `session_id`| `"default"` | Identifier for the session (used in registry) |

**Risk bands:**

| CPE range | Band       |
|-----------|------------|
| 0.0 – 1.0 | LOW        |
| 1.0 – 2.0 | MODERATE   |
| 2.0 – 3.0 | HIGH       |
| 3.0+      | CRITICAL   |

**Hard-blocked entity types** (always redacted, regardless of CPE):
- `US_SSN`
- `CREDIT_CARD`
- `ACCOUNT_NUMBER`

---

## Supported entity types

| Category | Entities |
|----------|----------|
| Personal | Person name, Location, Organization, Email, Phone, Date of birth, SSN, Medical condition, Credit card, IP address, Age, Salary, Ethnicity, Account number |
| Corporate | Financial amount, Financial metric, Internal projection, Confidential data |

---

## Development

```bash
git clone https://github.com/aman-panjwani/camp
cd camp
pip install -e ".[dev]"
python -m spacy download en_core_web_lg

# Run tests (no spaCy model required for unit tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=camp --cov-report=term-missing
```

---

## Citation

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

MIT © 2026 Aman Panjwani

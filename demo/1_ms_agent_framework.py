"""
pip install campii[agent-framework] python-dotenv
python -m spacy download en_core_web_lg
"""
import asyncio
import sys

from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.anthropic import AnthropicClient

from camp import CAMPMasker

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

CYAN, YELLOW, MAGENTA, GREEN, BOLD, RESET = (
    "\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[1m", "\033[0m"
)

masker = CAMPMasker(threshold=2.0, redaction_map={"US_SSN": "[BLOCKED]"})

message = (
    "Hi, I'm Sarah Johnson. Email: sarah.johnson@email.com, phone: 415-555-0192. "
    "My SSN is 512-34-7891. Please write a complaint email to Pacific Life Insurance."
)


async def main() -> None:
    result = masker.process_turn(message, turn_index=0)

    agent = Agent(
        client=AnthropicClient(model="claude-sonnet-4-6"),
        instructions="You are a helpful assistant. Follow user instructions precisely.",
    )

    response = await agent.run(result.sent_to_llm)
    raw   = response.text
    final = masker.demask_response(raw)

    print(f"\n{BOLD}Decision: {result.decision}  |  CPE: {result.cpe_score:.2f}{RESET}\n")
    print(f"{CYAN}{BOLD}RECEIVED{RESET}\n{CYAN}{message}{RESET}\n")
    print(f"{YELLOW}{BOLD}SENT TO AGENT  (PII masked){RESET}\n{YELLOW}{result.sent_to_llm}{RESET}\n")
    print(f"{MAGENTA}{BOLD}AGENT RESPONSE  (fake values){RESET}\n{MAGENTA}{raw}{RESET}\n")
    print(f"{GREEN}{BOLD}FINAL OUTPUT  (real values restored){RESET}\n{GREEN}{final}{RESET}")

    masker.reset()


asyncio.run(main())

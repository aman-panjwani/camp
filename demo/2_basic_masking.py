"""
pip install campii
"""
import os
import sys

import anthropic
from dotenv import load_dotenv

from camp import CAMPMasker

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

CYAN, YELLOW, MAGENTA, GREEN, BOLD, DIM, RESET = (
    "\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[1m", "\033[2m", "\033[0m"
)

masker = CAMPMasker(threshold=2.0, redaction_map={"US_SSN": "[BLOCKED]"})
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = (
    "Hi, I'm Sarah Johnson. Email: sarah.johnson@email.com, phone: 415-555-0192. "
    "My SSN is 512-34-7891. Please write a complaint email to Pacific Life Insurance."
)

result = masker.process_turn(message, turn_index=0)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": result.sent_to_llm}],
)
raw   = response.content[0].text
final = masker.demask_response(raw)

print(f"\n{BOLD}Decision: {result.decision}  |  CPE: {result.cpe_score:.2f}{RESET}\n")
print(f"{CYAN}{BOLD}RECEIVED{RESET}\n{CYAN}{message}{RESET}\n")
print(f"{YELLOW}{BOLD}SENT TO LLM  (PII masked){RESET}\n{YELLOW}{result.sent_to_llm}{RESET}\n")
print(f"{MAGENTA}{BOLD}LLM RESPONSE  (fake values){RESET}\n{MAGENTA}{raw}{RESET}\n")
print(f"{GREEN}{BOLD}FINAL OUTPUT  (real values restored){RESET}\n{GREEN}{final}{RESET}")

masker.reset()

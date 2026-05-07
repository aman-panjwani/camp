"""
pip install campii anthropic python-dotenv
python -m spacy download en_core_web_lg
"""
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

from camp import CAMPMasker

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

CYAN, YELLOW, MAGENTA, GREEN, BLUE, BOLD, DIM, RESET = (
    "\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[94m", "\033[1m", "\033[2m", "\033[0m"
)


def check_claim(claim_id: str, customer_name: str) -> dict:
    return {
        "claim_id":             claim_id,
        "customer":             customer_name,
        "status":               "Under Review",
        "filed_date":           "2024-10-15",
        "estimated_resolution": "2025-02-01",
        "assigned_to":          "Claims Team B",
        "notes":                "Additional documentation requested.",
    }


TOOLS = [
    {
        "name": "check_claim",
        "description": "Look up an insurance claim by claim ID and customer name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_id":      {"type": "string", "description": "The claim reference number"},
                "customer_name": {"type": "string", "description": "Full name of the customer"},
            },
            "required": ["claim_id", "customer_name"],
        },
    }
]

TOOL_REGISTRY = {"check_claim": check_claim}

masker = CAMPMasker(threshold=2.0, alpha=0.3, redaction_map={"US_SSN": "[BLOCKED]"})
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = (
    "Hi, I'm Sarah Johnson (sarah.johnson@email.com, 415-555-0192). "
    "Can you check the status of my insurance claim #CLM-789012 "
    "with Pacific Life Insurance?"
)

result = masker.process_turn(message, turn_index=0)

print(f"\n{BOLD}Decision: {result.decision}  |  CPE: {result.cpe_score:.2f}{RESET}\n")
print(f"{CYAN}{BOLD}RECEIVED (real PII){RESET}\n{CYAN}{message}{RESET}\n")
print(f"{YELLOW}{BOLD}SENT TO LLM (masked){RESET}\n{YELLOW}{result.sent_to_llm}{RESET}\n")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    tools=TOOLS,
    messages=[{"role": "user", "content": result.sent_to_llm}],
)

tool_results = []
for block in response.content:
    if block.type == "tool_use":
        print(f"{BLUE}{BOLD}TOOL CALL  ->  {block.name}{RESET}")
        print(f"{YELLOW}Args from LLM       (fake) : {json.dumps(block.input)}{RESET}")

        real_args = masker.demask_args(block.input)
        print(f"{GREEN}Args after demask    (real) : {json.dumps(real_args)}{RESET}")

        real_output = TOOL_REGISTRY[block.name](**real_args)
        print(f"{GREEN}Tool result          (real) : {json.dumps(real_output)}{RESET}")

        masked_output = masker.mask_content(real_output)
        print(f"{YELLOW}Tool result re-masked (fake) : {masked_output}{RESET}\n")

        tool_results.append(masker.build_tool_result(block.id, masked_output))

final_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    tools=TOOLS,
    messages=[
        {"role": "user",      "content": result.sent_to_llm},
        {"role": "assistant", "content": response.content},
        {"role": "user",      "content": tool_results},
    ],
)
raw   = final_response.content[0].text
final = masker.demask_response(raw)

print(f"{MAGENTA}{BOLD}LLM RESPONSE (fake values){RESET}\n{MAGENTA}{raw}{RESET}\n")
print(f"{GREEN}{BOLD}FINAL OUTPUT (real values restored){RESET}\n{GREEN}{final}{RESET}")

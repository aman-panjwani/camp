"""
pip install campii
python -m spacy download en_core_web_lg
"""
import sys

from camp import CAMPMasker

sys.stdout.reconfigure(encoding="utf-8")

CYAN, YELLOW, RED, GREEN, BOLD, DIM, RESET = (
    "\033[96m", "\033[93m", "\033[91m", "\033[92m", "\033[1m", "\033[2m", "\033[0m"
)

COLORS = {"PASS": GREEN, "BLOCK": RED, "PSEUDONYMIZE": YELLOW}

cases = [
    {
        "label": "PASS — no PII detected, score stays below threshold",
        "message": "Hey, can you summarise the key points from our last meeting?",
        "masker": CAMPMasker(threshold=2.0),
    },
    {
        "label": "BLOCK — hard-blocked entity found (SSN), sent straight to [BLOCKED]",
        "message": "My SSN is 512-34-7891. Can you look up my file?",
        "masker": CAMPMasker(threshold=99.0, redaction_map={"US_SSN": "[BLOCKED]"}),
    },
    {
        "label": "PSEUDONYMIZE — CPE crosses threshold, all PII replaced with fakes",
        "message": (
            "Hi, I'm Sarah Johnson. Email: sarah.johnson@email.com, "
            "phone: 415-555-0192. Please write a complaint email to Pacific Life Insurance."
        ),
        "masker": CAMPMasker(threshold=2.0),
    },
]

print()
for case in cases:
    result = case["masker"].process_turn(case["message"], turn_index=0)
    c = COLORS[result.decision]
    print(f"{c}{BOLD}[{result.decision}]{RESET}  {DIM}{case['label']}{RESET}")
    print(f"  {DIM}IN :{RESET}  {case['message']}")
    print(f"  {DIM}OUT:{RESET}  {c}{result.sent_to_llm}{RESET}")
    print(f"  {DIM}CPE: {result.cpe_score:.2f}  |  entities: {len(result.entities)}{RESET}")
    print()

"""
pip install campii
python -m spacy download en_core_web_lg
"""
import sys

from camp import CAMPMasker

sys.stdout.reconfigure(encoding="utf-8")

CYAN, YELLOW, GREEN, BOLD, DIM, RESET = (
    "\033[96m", "\033[93m", "\033[92m", "\033[1m", "\033[2m", "\033[0m"
)

custom_patterns = [
    {"entity": "EMPLOYEE_ID",  "pattern": r"\bEMP-\d{6}\b",          "score": 0.9},
    {"entity": "INTERNAL_REF", "pattern": r"\bPROJ-[A-Z]{2}\d{4}\b", "score": 0.9},
]

masker = CAMPMasker(
    threshold=2.0,
    redaction_map={"EMPLOYEE_ID": "[EMP-REDACTED]"},
    custom_patterns=custom_patterns,
)

message = (
    "Hi, I'm David Lee (EMP-004821). I'm working on project PROJ-AL2024. "
    "Please share the report with david.lee@company.com."
)

result = masker.process_turn(message, turn_index=0)

print(f"\n{BOLD}Decision: {result.decision}  |  CPE: {result.cpe_score:.2f}{RESET}\n")

print(f"{BOLD}Detected entities:{RESET}")
for e in result.entities:
    print(f"  {YELLOW}{e.label():<18}{RESET} {e.value}")

print(f"\n{CYAN}{BOLD}RECEIVED{RESET}")
print(f"{CYAN}{message}{RESET}\n")

print(f"{YELLOW}{BOLD}SENT TO LLM{RESET}")
print(f"{YELLOW}{result.sent_to_llm}{RESET}")

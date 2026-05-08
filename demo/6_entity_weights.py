"""
pip install campii
python -m spacy download en_core_web_lg
"""
import sys

from camp import CAMPMasker

sys.stdout.reconfigure(encoding="utf-8")

YELLOW, GREEN, RED, CYAN, BOLD, DIM, RESET = (
    "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[1m", "\033[2m", "\033[0m"
)

message = "Hi, I'm Sarah Johnson (sarah.johnson@email.com, 415-555-0192)."

r1 = CAMPMasker(threshold=4.0).process_turn(message, turn_index=0)
r2 = CAMPMasker(
    threshold=4.0,
    entity_weights={"PERSON": 1.0, "EMAIL_ADDRESS": 1.0, "PHONE_NUMBER": 1.0},
).process_turn(message, turn_index=0)

print(f"\n{BOLD}Message:{RESET} {DIM}{message}{RESET}\n")
print(f"  {'Profile':<20}  {'CPE':>6}  Decision")
print(f"  {'─' * 40}")
print(f"  {CYAN}{'Default':<20}{RESET}  {r1.cpe_score:>6.2f}  {YELLOW}{BOLD}{r1.decision}{RESET}")
print(f"  {CYAN}{'Healthcare/Finance':<20}{RESET}  {r2.cpe_score:>6.2f}  {RED}{BOLD}{r2.decision}{RESET}\n")

"""Shared fixtures for the CAMP test suite."""
import pytest

from camp.core.entities import PERSON, LOCATION, ORGANIZATION, EMAIL, SSN, SALARY, ACCOUNT
from camp.core.extractor import DetectedEntity
from camp.core.masker import CAMPMasker
from camp.integrations.llm import CAMPSession

# ── Sample conversations ──────────────────────────────────────────

FINANCE_TURNS = [
    "Hi I need help with my bank account.",
    "My name is Michael Torres.",
    "I bank with Chase, account ending in 4872.",
    "I live in Austin, Texas.",
    "My income is around 92,000 dollars a year.",
    "My address is 412 Riverside Drive Austin TX 78701.",
    "My SSN is 512-34-7891.",
    "Will this affect my mortgage application?",
]

HEALTHCARE_TURNS = [
    "Hi I need help with my insurance claim.",
    "My name is Sarah Johnson.",
    "I live in Denver, Colorado.",
    "I was born on March 5, 1988.",
    "My policy number is LM-4492817.",
    "You can reach me at sarah.johnson88@gmail.com.",
]

CLEAN_TURNS = [
    "What's the weather like today?",
    "Can you recommend a good book?",
    "Tell me a joke.",
]


# ── Entity helpers ────────────────────────────────────────────────

def make_entity(etype: str, value: str, turn: int = 0, score: float = 0.85) -> DetectedEntity:
    return DetectedEntity(entity_type=etype, value=value, score=score, turn_index=turn)


# ── Entity fixtures ───────────────────────────────────────────────

@pytest.fixture
def person_entity() -> DetectedEntity:
    return make_entity(PERSON, "Michael Torres")

@pytest.fixture
def ssn_entity() -> DetectedEntity:
    return make_entity(SSN, "512-34-7891")

@pytest.fixture
def salary_entity() -> DetectedEntity:
    return make_entity(SALARY, "92,000 dollars a year")

@pytest.fixture
def location_entity() -> DetectedEntity:
    return make_entity(LOCATION, "Austin, Texas")


# ── Masker fixtures ───────────────────────────────────────────────

@pytest.fixture
def masker() -> CAMPMasker:
    return CAMPMasker(threshold=2.0, alpha=0.3)

@pytest.fixture
def low_threshold_masker() -> CAMPMasker:
    return CAMPMasker(threshold=0.3, alpha=0.3)

@pytest.fixture
def camp_session() -> CAMPSession:
    return CAMPSession(threshold=2.0, alpha=0.3)

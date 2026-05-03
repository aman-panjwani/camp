# camp.core.entities
# ------------------
# Central definition of all PII entity types.
# ENTITY_CATALOG is the single source of truth for name, label, and weight.
# ENTITY_WEIGHTS / ENTITY_LABELS / ALL_ENTITY_TYPES are derived from it so
# all downstream code continues to work without changes.

from dataclasses import dataclass
from typing import List

# ── Entity type constants ─────────────────────────────────────────────────────
# Module-level strings kept so all existing import sites remain unchanged.

# Personal PII
PERSON         = "PERSON"
LOCATION       = "LOCATION"
ORGANIZATION   = "ORGANIZATION"
EMAIL          = "EMAIL_ADDRESS"
PHONE          = "PHONE_NUMBER"
DATE_OF_BIRTH  = "DATE_TIME"
SSN            = "US_SSN"
MEDICAL        = "MEDICAL"
CREDIT_CARD    = "CREDIT_CARD"
IP_ADDRESS     = "IP_ADDRESS"
AGE            = "AGE"
SALARY         = "SALARY"
ETHNICITY      = "ETHNICITY"
ACCOUNT        = "ACCOUNT_NUMBER"
DRIVER_LICENSE = "US_DRIVER_LICENSE"

# Corporate sensitive
FINANCIAL_AMOUNT    = "FINANCIAL_AMOUNT"
FINANCIAL_METRIC    = "FINANCIAL_METRIC"
INTERNAL_PROJECTION = "INTERNAL_PROJECTION"
CONFIDENTIAL_DATA   = "CONFIDENTIAL_DATA"

# Financial / transactional (new)
SWIFT_BIC      = "SWIFT_BIC"
IBAN           = "IBAN_CODE"       # Presidio native
TRANSACTION_ID = "TRANSACTION_ID"
CRYPTO         = "CRYPTO"          # Presidio native
US_ITIN        = "US_ITIN"         # Presidio native


# ── EntityDef: single source of truth ────────────────────────────────────────

@dataclass(frozen=True)
class EntityDef:
    """Defines a PII entity type with its display label and CPE scoring weight."""
    name:   str    # entity type string - matches Presidio entity type
    label:  str    # human-readable label shown in output
    weight: float  # CPE scoring weight in (0, 1] - higher = more re-id risk


ENTITY_CATALOG: List[EntityDef] = [
    # ── Hard financial / identity ─────────────────────────────────────────────
    EntityDef(SSN,            "SSN",               1.00),
    EntityDef(CREDIT_CARD,    "Credit Card",        1.00),
    EntityDef(IBAN,           "IBAN",               1.00),
    EntityDef(US_ITIN,        "US ITIN",            1.00),
    EntityDef(SWIFT_BIC,      "SWIFT/BIC Code",     1.00),
    EntityDef(ACCOUNT,        "Account Number",     0.95),
    EntityDef(CONFIDENTIAL_DATA, "Confidential Data", 0.95),
    EntityDef(CRYPTO,         "Crypto Wallet",      0.90),
    EntityDef(INTERNAL_PROJECTION, "Internal Projection", 0.90),
    EntityDef(DATE_OF_BIRTH,  "Date of Birth",      0.90),
    EntityDef(DRIVER_LICENSE, "Driver License",     0.85),
    EntityDef(FINANCIAL_METRIC, "Financial Metric", 0.85),
    EntityDef(MEDICAL,        "Medical Condition",  0.85),
    EntityDef(EMAIL,          "Email Address",      0.80),
    EntityDef(FINANCIAL_AMOUNT, "Financial Amount", 0.80),
    EntityDef(PHONE,          "Phone Number",       0.75),
    EntityDef(ETHNICITY,      "Ethnicity",          0.70),
    EntityDef(TRANSACTION_ID, "Transaction ID",     0.70),
    EntityDef(PERSON,         "Person Name",        0.60),
    EntityDef(SALARY,         "Salary",             0.60),
    EntityDef(AGE,            "Age",                0.55),
    EntityDef(LOCATION,       "Location",           0.50),
    EntityDef(IP_ADDRESS,     "IP Address",         0.50),
    EntityDef(ORGANIZATION,   "Organization",       0.30),
]

# ── Derived lookups (backward-compatible with all existing code) ──────────────

ENTITY_WEIGHTS:   dict[str, float] = {e.name: e.weight for e in ENTITY_CATALOG}
ENTITY_LABELS:    dict[str, str]   = {e.name: e.label  for e in ENTITY_CATALOG}
ALL_ENTITY_TYPES: set[str]         = {e.name for e in ENTITY_CATALOG}


# ── Default redaction map ─────────────────────────────────────────────────────
# Maps entity_type → replacement string for entities always redacted regardless
# of CPE score.  Pass redaction_map= to any CAMP class to override.

DEFAULT_REDACTION_MAP: dict[str, str] = {
    SSN:         "[BLOCKED]",
    CREDIT_CARD: "[BLOCKED]",
    ACCOUNT:     "[BLOCKED]",
}

HARD_BLOCK_TYPES: set[str] = set(DEFAULT_REDACTION_MAP)


# ── Risk bands ────────────────────────────────────────────────────────────────

RISK_BANDS: list[tuple[float, float, str, str]] = [
    (0.0,  1.0,  "LOW",      "\033[92m"),
    (1.0,  2.0,  "MODERATE", "\033[93m"),
    (2.0,  3.0,  "HIGH",     "\033[91m"),
    (3.0,  999,  "CRITICAL", "\033[95m"),
]

RESET = "\033[0m"


def get_risk_band(cpe: float) -> str:
    for low, high, band, _ in RISK_BANDS:
        if low <= cpe < high:
            return band
    return "CRITICAL"


def get_risk_color(cpe: float) -> str:
    for low, high, _, color in RISK_BANDS:
        if low <= cpe < high:
            return color
    return "\033[95m"

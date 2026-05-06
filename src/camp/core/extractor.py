# camp.core.extractor
# -------------------
# Per-turn PII extractor using Microsoft Presidio + spaCy.
# Custom recognizers for:
#   Personal : Salary, Account Number, Street Address, SWIFT/BIC, Transaction ID
#   Corporate: Financial Amount, Financial Metric,
#              Internal Projection, Confidential Data

import re
from dataclasses import dataclass
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from camp.core.entities import (
    ACCOUNT,
    ALL_ENTITY_TYPES,
    CONFIDENTIAL_DATA,
    CREDIT_CARD,
    DEFAULT_REDACTION_MAP,
    DRIVER_LICENSE,
    ENTITY_LABELS,
    ENTITY_WEIGHTS,
    FINANCIAL_AMOUNT,
    FINANCIAL_METRIC,
    INTERNAL_PROJECTION,
    LOCATION,
    ORGANIZATION,
    SALARY,
    SWIFT_BIC,
    TRANSACTION_ID,
)

# Normalize Presidio built-in types that don't match our entity constants.
# US_BANK_NUMBER → ACCOUNT_NUMBER so routing/account numbers get hard-blocked.
# US_DRIVER_LICENSE → our DRIVER_LICENSE constant so it gets pseudonymized.
_PRESIDIO_TO_CAMP: dict[str, str] = {
    "US_BANK_NUMBER":    ACCOUNT,
    "US_DRIVER_LICENSE": DRIVER_LICENSE,
}


@dataclass
class DetectedEntity:
    entity_type: str
    value:       str
    score:       float
    turn_index:  int

    def weight(self) -> float:
        return ENTITY_WEIGHTS.get(self.entity_type, 0.3)

    def label(self) -> str:
        return ENTITY_LABELS.get(self.entity_type, self.entity_type)

    def is_hard_block(self, redaction_map: dict[str, str] | None = None) -> bool:
        m = redaction_map if redaction_map is not None else DEFAULT_REDACTION_MAP
        return self.entity_type in m

    def redaction_value(self, redaction_map: dict[str, str] | None = None) -> str:
        m = redaction_map if redaction_map is not None else DEFAULT_REDACTION_MAP
        return m.get(self.entity_type, "[BLOCKED]")

    def __repr__(self) -> str:
        return (
            f"[{self.label()}] '{self.value}' "
            f"(score={self.score:.2f}, turn={self.turn_index})"
        )


def _build_analyzer() -> AnalyzerEngine:
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider   = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )

    # ── Personal PII recognizers ──────────────────────────────

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=SALARY,
        patterns=[
            # "$8,750 per month" / "$80,000 per year"
            Pattern(
                "SALARY_DOLLAR_PERIOD",
                r"\$\d{1,3}(?:,\d{3})+\s*per\s+(?:month|year|hour|week|annum)",
                0.85,
            ),
            # "80,000 dollars" / "100k"
            Pattern(
                "SALARY_AMOUNT_WORDS",
                r"\b\d{2,3}[,.]?\d{3}\s*(?:dollars?|USD|k\b)"
                r"(?:\s*(?:per year|a year|annually))?",
                0.80,
            ),
            # "earning $8,750" / "salary of $120,000"
            Pattern(
                "SALARY_CONTEXT",
                r"(?:income|salary|earn\w*|compensation|make|paid)\s+"
                r"(?:is\s+|around\s+|about\s+|of\s+)?"
                r"\$?\d{1,3}[,.]?\d{3}",
                0.80,
            ),
        ],
        context=["salary", "compensation", "earn", "income", "paid", "wage",
                 "annual", "monthly", "per year", "per month"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=ACCOUNT,
        patterns=[
            # "account number 00987654321"
            Pattern(
                "ACCOUNT_NUMBER_FULL",
                r"\baccount\s+number\s+\d{5,20}\b",
                0.90,
            ),
            # "account ending in 4872"
            Pattern(
                "ACCOUNT_ENDING",
                r"\baccount\s+(?:number\s+)?(?:ending\s+(?:in\s+)?)?\d{4}\b",
                0.85,
            ),
            # "routing number 071000013"
            Pattern(
                "ROUTING_NUMBER",
                r"\brouting\s+(?:number\s+)?\d{9}\b",
                0.90,
            ),
        ],
        context=["account", "routing", "bank", "checking", "savings",
                 "aba", "wire", "ach", "deposit"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=CREDIT_CARD,
        patterns=[
            # "credit card ending with 8891"
            Pattern(
                "CARD_ENDING_PHRASE",
                r"\b(?:credit\s+|debit\s+)?card\s+(?:number\s+)?"
                r"(?:ending\s+(?:with\s+|in\s+))?\d{4}\b",
                0.85,
            ),
        ],
        context=["card", "credit", "debit", "charge", "transaction"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=LOCATION,
        patterns=[
            Pattern(
                "STREET_ADDRESS",
                r"\b\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*"
                r"\s+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|"
                r"Boulevard|Blvd|Lane|Ln|Court|Ct|Way|Place|Pl)"
                r"(?:\s+[A-Z][a-zA-Z]+(?:,?\s+[A-Z]{2})?"
                r"(?:\s+\d{5})?)?",
                0.85,
            ),
        ],
    ))

    # ── Corporate sensitive recognizers ───────────────────────

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=ORGANIZATION,
        patterns=[
            # Catches "Chase Bank", "GreenField Investments" when spaCy NER misses them.
            # "Financial" excluded - too generic (matches "Senior Financial").
            Pattern(
                "ORG_SUFFIX",
                r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+"
                r"(?:Bank|Investments?|Capital|Fund|Securities|"
                r"Insurance|Trust|Group|Holdings?|Corp(?:oration)?|"
                r"Inc(?:orporated)?|Ltd|LLC|LLP|Partners?|Advisors?)\b",
                0.72,
            ),
        ],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=FINANCIAL_AMOUNT,
        patterns=[
            Pattern(
                "DEAL_SIZE",
                r"\$\d+(?:\.\d+)?(?:M|B|K|million|billion|thousand)"
                r"(?:\s*(?:deal|acquisition|valuation|transaction))?",
                0.85,
            ),
            Pattern(
                "AMOUNT_WORDS",
                r"\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)"
                r"\s*(?:dollar|USD)?",
                0.80,
            ),
            # Plain dollar amounts like "$2,340" - low score so salary patterns take priority.
            Pattern(
                "DOLLAR_AMOUNT",
                r"\$\d{1,3}(?:,\d{3})+(?:\.\d{2})?",
                0.65,
            ),
        ],
        context=["payment", "amount", "transaction", "transfer", "deal",
                 "cost", "fee", "value", "price", "fund"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=FINANCIAL_METRIC,
        patterns=[
            Pattern(
                "PERCENTAGE_METRIC",
                r"\b\d+(?:\.\d+)?%\s*"
                r"(?:increase|decrease|growth|decline|haircut|"
                r"default|rate|margin|return|yield|loss|gain)?",
                0.80,
            ),
            Pattern(
                "METRIC_CONTEXT",
                r"(?:default rates?|growth rates?|margins?|yields?|"
                r"haircut|synergies)\s+(?:of\s+|at\s+|around\s+)?"
                r"\d+(?:\.\d+)?%",
                0.85,
            ),
        ],
        context=["rate", "margin", "yield", "return", "growth",
                 "decline", "default", "haircut", "synerg"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=INTERNAL_PROJECTION,
        patterns=[
            Pattern(
                "MODELING_STATEMENT",
                r"(?:internally|we are|we're|modeling|projecting|"
                r"forecasting|estimating)\s+"
                r"(?:a\s+)?(?:~|approximately|around|about)?\s*"
                r"\d+(?:\.\d+)?%",
                0.85,
            ),
            Pattern(
                "DOWNSIDE_SCENARIO",
                r"(?:downside|upside|base case|worst case|best case)"
                r"\s+(?:scenario|case|projection|estimate|model)",
                0.80,
            ),
        ],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=CONFIDENTIAL_DATA,
        patterns=[
            Pattern(
                "NOT_DISCLOSED",
                r"(?:not|hasn't|have not|hasn't been)\s+"
                r"(?:been\s+)?(?:disclosed|shared|released|published|"
                r"announced|made public)\s*(?:externally|publicly|yet)?",
                0.90,
            ),
            Pattern(
                "INTERNAL_ONLY",
                r"(?:internal(?:ly)?|confidential(?:ly)?|"
                r"proprietary|non-public|undisclosed)\s+"
                r"(?:data|information|figures?|numbers?|projections?|"
                r"estimates?|analysis|report|we(?:'re|\s+are)\s+\w+ing)",
                0.85,
            ),
        ],
    ))

    # ── New: Financial / transactional recognizers ────────────

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=SWIFT_BIC,
        # global_regex_flags removes IGNORECASE - SWIFT codes are always uppercase,
        # and without this [A-Z]{8} matches any 8-letter English word case-insensitively.
        global_regex_flags=re.DOTALL | re.MULTILINE,
        patterns=[
            # 8-char SWIFT: 4-letter bank code + 2-letter country + 2 alphanumeric location
            Pattern(
                "SWIFT_8",
                r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}\b",
                0.60,
            ),
            # 11-char SWIFT: same + 3-char branch code
            Pattern(
                "SWIFT_11",
                r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}[A-Z0-9]{3}\b",
                0.75,
            ),
        ],
        context=["swift", "bic", "wire", "transfer", "correspondent",
                 "bank", "international", "remittance"],
    ))

    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity=TRANSACTION_ID,
        patterns=[
            # Standard UUID
            Pattern(
                "UUID",
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
                0.90,
            ),
            # Prefixed IDs: TXN-45892173, PAY-ABC123, REF-XYZ
            Pattern(
                "TXN_PREFIX",
                r"\b(?:TXN|TRX|REF|TID|PAY|ORD|INV|PMT)[_\-][A-Z0-9]{4,20}\b",
                0.90,
            ),
        ],
        context=["transaction", "reference", "order", "payment",
                 "invoice", "receipt", "confirmation", "id"],
    ))

    return analyzer


_analyzer: AnalyzerEngine | None = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = _build_analyzer()
    return _analyzer


def _merge_locations(
    text: str,
    entities: list[DetectedEntity],
) -> list[DetectedEntity]:
    """Merge consecutive LOCATION entities separated by whitespace or comma."""
    loc_entities = [e for e in entities if e.entity_type == LOCATION]
    non_loc      = [e for e in entities if e.entity_type != LOCATION]

    merged: list[DetectedEntity] = []
    i = 0
    while i < len(loc_entities):
        current = loc_entities[i]

        if i + 1 < len(loc_entities):
            nxt      = loc_entities[i + 1]
            curr_pos = text.lower().find(current.value.lower())
            next_pos = text.lower().find(nxt.value.lower(), curr_pos + 1)

            if curr_pos != -1 and next_pos != -1:
                between = text[curr_pos + len(current.value):next_pos]
                if re.fullmatch(r'[\s,]+', between):
                    merged.append(DetectedEntity(
                        entity_type=LOCATION,
                        value=text[curr_pos:next_pos + len(nxt.value)].strip(),
                        score=max(current.score, nxt.score),
                        turn_index=current.turn_index,
                    ))
                    i += 2
                    continue

        merged.append(current)
        i += 1

    return non_loc + merged


def extract_pii(
    text: str,
    turn_index: int = 0,
    custom_patterns: list[dict[str, Any]] | None = None,
) -> list[DetectedEntity]:
    """Extract PII entities from text using Presidio + custom recognizers.

    custom_patterns: optional list of dicts, each with keys:
        entity  – entity type name (str, e.g. "EMPLOYEE_ID")
        pattern – regex string
        score   – confidence score (float, default 0.8)
    """
    ad_hoc = []
    custom_types: set[str] = set()
    if custom_patterns:
        for p in custom_patterns:
            entity  = p["entity"]
            pattern = p["pattern"]
            score   = p.get("score", 0.8)
            ad_hoc.append(PatternRecognizer(
                supported_entity=entity,
                patterns=[Pattern(entity, pattern, score)],
            ))
            custom_types.add(entity)

    results = get_analyzer().analyze(
        text=text, language="en", score_threshold=0.35,
        ad_hoc_recognizers=ad_hoc or None,
    )
    entities: list[DetectedEntity] = []
    seen:     set[tuple[str, str]] = set()
    covered:  list[tuple[int, int]] = []

    for r in sorted(results, key=lambda r: r.end - r.start, reverse=True):
        value = text[r.start:r.end].strip()

        if not value:
            continue

        # Skip pure 4-digit numbers (Presidio date false positive)
        if re.fullmatch(r'\d{4}', value):
            continue

        if any(r.start >= cs and r.end <= ce for cs, ce in covered):
            continue

        # Normalize Presidio-native types to our entity constants;
        # allow custom (caller-defined) types through without normalisation.
        entity_type = _PRESIDIO_TO_CAMP.get(r.entity_type, r.entity_type)
        if entity_type not in ALL_ENTITY_TYPES and entity_type not in custom_types:
            continue

        # spaCy NER sometimes extends ORGANIZATION spans to include surrounding context
        # (e.g. "checking and savings account with Chase Bank").
        # When that happens, try to salvage the clean company name via ORG_SUFFIX regex;
        # skip only if no company name can be found within the span.
        if entity_type == ORGANIZATION:
            words = value.split()
            if any(w.islower() and len(w) > 1 for w in words[1:]):
                _ORG_SUFFIX_RE = (
                    r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+'
                    r'(?:Bank|Investments?|Capital|Fund|Securities|'
                    r'Insurance|Trust|Group|Holdings?|Corp(?:oration)?|'
                    r'Inc(?:orporated)?|Ltd|LLC|LLP|Partners?|Advisors?)\b'
                )
                m = re.search(_ORG_SUFFIX_RE, value)
                if m:
                    value = m.group(0)
                else:
                    continue

        key = (entity_type, value.lower())
        if key in seen:
            continue

        seen.add(key)
        covered.append((r.start, r.end))
        entities.append(DetectedEntity(
            entity_type=entity_type,
            value=value,
            score=r.score,
            turn_index=turn_index,
        ))

    return _merge_locations(text, entities)


def mask_text(
    text: str,
    entities: list[DetectedEntity],
    redaction_map: dict[str, str] | None = None,
) -> str:
    """Replace detected PII with per-entity redaction strings or type placeholders."""
    if not entities:
        return text

    masked = text
    for entity in sorted(entities, key=lambda e: len(e.value), reverse=True):
        placeholder = (
            entity.redaction_value(redaction_map) if entity.is_hard_block(redaction_map)
            else f"[{entity.label()}]"
        )
        masked = re.sub(
            re.escape(entity.value),
            placeholder,
            masked,
            flags=re.IGNORECASE,
        )
    return masked

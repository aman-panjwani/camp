# camp.core.pseudonymizer
# -----------------------
# Generates consistent synthetic substitutes for real PII values.
# Same real value always maps to the same synthetic value within a session.
# Ensures referential consistency across turns so the LLM receives
# a coherent conversation with no real PII.

import random
import re
from typing import TYPE_CHECKING

from faker import Faker

if TYPE_CHECKING:
    from camp.core.registry import TurnRecord

from camp.core.entities import (
    ACCOUNT,
    AGE,
    CONFIDENTIAL_DATA,
    CREDIT_CARD,
    DATE_OF_BIRTH,
    DRIVER_LICENSE,
    EMAIL,
    ENTITY_LABELS,
    ETHNICITY,
    FINANCIAL_AMOUNT,
    FINANCIAL_METRIC,
    INTERNAL_PROJECTION,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    SALARY,
    SWIFT_BIC,
    TRANSACTION_ID,
)
from camp.core.extractor import DetectedEntity

fake = Faker()
Faker.seed(42)  # Reproducible results for paper experiments


class Pseudonymizer:
    """
    Session-level pseudonym map.
    Generates consistent fake substitutes and provides reverse mapping
    for de-masking LLM responses before returning them to the user.
    """

    def __init__(self, seed: int | None = 42, redaction_map: dict[str, str] | None = None) -> None:
        if seed is not None:
            Faker.seed(seed)
        self._redaction_map = redaction_map  # None → use DEFAULT_REDACTION_MAP
        self._map:         dict[str, str] = {}
        self._reverse_map: dict[str, str] = {}

    def get_pseudonym(self, entity: DetectedEntity) -> str:
        """
        Returns a consistent synthetic substitute for a real value.
        Hard-blocked entities always return [BLOCKED].
        """
        if entity.is_hard_block(self._redaction_map):
            return entity.redaction_value(self._redaction_map)

        real = entity.value.strip()

        if real in self._map:
            return self._map[real]

        pseudo = self._generate(entity.entity_type, real)
        self._map[real]           = pseudo
        self._reverse_map[pseudo] = real

        # For person names, also register first-name so the LLM can address
        # the user informally (e.g. "Hi Javier") and still get demasked.
        if entity.entity_type == PERSON:
            real_parts   = real.split()
            pseudo_parts = pseudo.split()
            if len(real_parts) >= 2 and len(pseudo_parts) >= 2:
                fake_first = pseudo_parts[0]
                real_first = real_parts[0]
                if fake_first not in self._reverse_map:
                    self._reverse_map[fake_first] = real_first

        return pseudo

    def _generate(self, entity_type: str, real: str) -> str:
        """Generate a realistic fake value for the given entity type."""

        if entity_type == PERSON:
            return fake.name()

        if entity_type == LOCATION:
            if re.search(r'\d+\s+\w+', real):
                return fake.address().replace('\n', ', ')
            return f"{fake.city()}, {fake.state_abbr()}"

        if entity_type == ORGANIZATION:
            return fake.company()

        if entity_type == EMAIL:
            return fake.email()

        if entity_type == PHONE:
            return fake.phone_number()

        if entity_type == DATE_OF_BIRTH:
            return fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%B %d, %Y")

        if entity_type == SALARY:
            numbers = re.findall(r'[\d,]+', real)
            if numbers:
                try:
                    amount:    float = float(int(numbers[0].replace(',', '')))
                    variation: float = float(random.randint(-10, 10))
                    new_amount = round(int(amount * (1 + variation / 100)) / 1000) * 1000
                    return f"{new_amount:,} dollars a year"
                except ValueError:
                    pass
            return f"{random.randint(60, 120) * 1000:,} dollars a year"

        if entity_type == AGE:
            numbers = re.findall(r'\d+', real)
            if numbers:
                age       = int(numbers[0])
                variation = random.randint(-3, 3)
                return str(max(18, age + variation))
            return str(random.randint(25, 55))

        if entity_type == DRIVER_LICENSE:
            return fake.bothify("?#######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        if entity_type == ETHNICITY:
            options = [
                "European descent",
                "Asian background",
                "diverse background",
                "multicultural background",
            ]
            return random.choice(options)

        if entity_type == FINANCIAL_AMOUNT:
            numbers = re.findall(r'[\d,]+', real)
            if numbers:
                try:
                    amount    = float(numbers[0].replace(',', ''))
                    variation = random.uniform(0.85, 1.15)
                    new_amount = amount * variation
                    suffix_match = re.search(r'[MBKmbk]$', real.strip())
                    if suffix_match:
                        return f"${round(new_amount)}{suffix_match.group(0).upper()}"
                    if amount >= 1_000_000:
                        return f"${new_amount / 1_000_000:.1f}M"
                    if amount >= 1_000:
                        return f"${round(new_amount):,}"
                    return f"${new_amount:.2f}"
                except ValueError:
                    pass
            return f"${random.randint(100, 500)}M"

        if entity_type == FINANCIAL_METRIC:
            numbers = re.findall(r'[\d.]+', real)
            if numbers:
                try:
                    pct       = float(numbers[0])
                    variation = random.uniform(-5, 5)
                    new_pct   = round(max(1.0, pct + variation), 1)
                    suffix    = re.sub(r'^[\d.]+%\s*', '', real).strip()
                    result    = f"{new_pct}%"
                    if suffix:
                        result += f" {suffix}"
                    return result
                except ValueError:
                    pass
            return f"{random.randint(5, 30)}%"

        if entity_type == INTERNAL_PROJECTION:
            numbers = re.findall(r'[\d.]+', real)
            if numbers:
                try:
                    pct       = float(numbers[0])
                    variation = random.uniform(-8, 8)
                    new_pct   = round(max(1.0, pct + variation), 1)
                    return re.sub(r'[\d.]+%', f"{new_pct}%", real, count=1)
                except ValueError:
                    pass
            return real

        if entity_type == CONFIDENTIAL_DATA:
            alternatives = [
                "has been reviewed internally",
                "is part of standard reporting",
                "has been noted in the analysis",
                "is under internal review",
                "is included in due diligence",
            ]
            return random.choice(alternatives)

        if entity_type == ACCOUNT:
            # Preserve label prefix ("account number", "routing number") and randomize digits
            return re.sub(
                r'\d+',
                lambda m: ''.join(str(random.randint(0, 9)) for _ in m.group()),
                real,
                count=1,
            )

        if entity_type == CREDIT_CARD:
            # Preserve phrase ("credit card ending with") and randomize the 4-digit suffix
            return re.sub(r'\d{4}', lambda _: str(random.randint(1000, 9999)), real)

        if entity_type == TRANSACTION_ID:
            # UUID format → new UUID
            if re.fullmatch(
                r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}'
                r'-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', real
            ):
                import uuid
                return str(uuid.uuid4())
            # PREFIX-NUMBER format (TXN-45892173, PAY-ABC123) - keep prefix, randomize suffix
            m = re.match(r'^([A-Z]{2,5}[_\-])([A-Z0-9]+)$', real, re.IGNORECASE)
            if m:
                prefix, suffix = m.group(1), m.group(2)
                fake_suffix = ''.join(
                    str(random.randint(0, 9)) if c.isdigit()
                    else random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    for c in suffix
                )
                return prefix + fake_suffix
            return real

        if entity_type == SWIFT_BIC:
            _LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            _ALNUM   = _LETTERS + '0123456789'
            _COUNTRIES = ['US', 'GB', 'DE', 'FR', 'JP', 'CH', 'AU', 'CA', 'SG', 'HK']
            bank    = ''.join(random.choices(_LETTERS, k=4))
            country = random.choice(_COUNTRIES)
            loc     = ''.join(random.choices(_ALNUM, k=2))
            if len(real) == 11:
                branch = ''.join(random.choices(_ALNUM, k=3))
                return bank + country + loc + branch
            return bank + country + loc

        return f"[{ENTITY_LABELS.get(entity_type, entity_type)}]"

    def pseudonymize_text(self, text: str, entities: list[DetectedEntity]) -> str:
        """Replace all PII values in text with pseudonyms."""
        if not entities:
            return text

        result = text
        for entity in sorted(entities, key=lambda e: len(e.value), reverse=True):
            pseudo = self.get_pseudonym(entity)
            result = re.sub(
                re.escape(entity.value),
                pseudo,
                result,
                flags=re.IGNORECASE,
            )
        return result

    def demask_response(self, response: str) -> str:
        """Restore original values in an LLM response using the reverse map."""
        result = response
        for pseudo, real in sorted(
            self._reverse_map.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            if pseudo in result:
                result = result.replace(pseudo, real)
        return result

    def rewrite_history(self, turns: "list[TurnRecord]") -> list[str]:
        """
        Retroactively rewrite full conversation history with pseudonyms.
        Called when CPE threshold is crossed.
        """
        rewritten = []
        for turn in turns:
            pseudo_text = self.pseudonymize_text(turn.raw_text, turn.entities)
            rewritten.append(pseudo_text)
        return rewritten

    def pseudonym_map(self) -> dict[str, str]:
        return dict(self._map)

    def reverse_map(self) -> dict[str, str]:
        return dict(self._reverse_map)

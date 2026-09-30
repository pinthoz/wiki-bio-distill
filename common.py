"""Shared code: schema, prompts, validation and metrics."""

import json
import re
import unicodedata

from pydantic import BaseModel, Field, ValidationError


class Biography(BaseModel):
    name: str = Field(description="Person's full name, as it appears in the text")
    birth_date: str | None = Field(
        None, description="YYYY-MM-DD, YYYY-MM or YYYY; null if missing"
    )
    birth_place: str | None = Field(
        None, description="City or town of birth; null if missing"
    )
    death_date: str | None = Field(
        None, description="YYYY-MM-DD, YYYY-MM or YYYY; null if alive or missing"
    )
    death_place: str | None = Field(
        None, description="City or town of death; null if missing"
    )
    nationality: list[str] = Field(
        default_factory=list, description="Adjectives, e.g. ['português']"
    )
    occupations: list[str] = Field(
        default_factory=list,
        description="Main professions or occupations, in the singular",
    )


SCHEMA = Biography.model_json_schema()
FIELDS = list(Biography.model_fields)

# Detailed prompt: used by the teacher and by the base model (fair baseline)
SYSTEM_TEACHER = """You are an information extraction system. You read the introduction of a \
Portuguese-language Wikipedia biography and return the requested data.

Rules:
- Use only information stated explicitly in the text. Never invent or infer.
- Dates in ISO format: YYYY-MM-DD; if only month and year are given, YYYY-MM; if only the year, YYYY.
- Missing fields: null (missing lists: []).
- Places: only the city or town, without the country (e.g. "Lisboa", not "Lisboa, Portugal").
- Nationality and occupations: in Portuguese, as in the text, lowercase, singular and masculine \
(e.g. "português", "escritor", "político")."""

# Short prompt: the student learns the behavior from the examples, not from long instructions
SYSTEM_STUDENT = "Extract the biography data as JSON."


def messages(text: str, system: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def parse_json(raw: str) -> dict | None:
    """Try to parse the generated JSON; tolerates surrounding text and ```json blocks."""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return Biography.model_validate(json.loads(m.group(0))).model_dump()
    except (json.JSONDecodeError, ValidationError):
        return None


# Metrics
def norm(value: str) -> str:
    """Normalize for comparison: lowercase, no accents, no punctuation at the ends."""
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip(" .,;")


def as_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {norm(v) for v in value if v}
    return {norm(str(value))}


def evaluate(preds: list[dict | None], golds: list[dict]) -> dict:
    """Per-field (micro) and overall precision, recall and F1; invalid JSON counts as empty."""
    stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    invalid = 0
    for pred, gold in zip(preds, golds):
        if pred is None:
            invalid += 1
            pred = {}
        for f in FIELDS:
            p, g = as_set(pred.get(f)), as_set(gold.get(f))
            stats[f]["tp"] += len(p & g)
            stats[f]["fp"] += len(
                p - g
            )  # includes hallucinations: a value where there should be null
            stats[f]["fn"] += len(g - p)

    def prf(s):
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 1.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    total = {k: sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")}
    return {
        "n": len(golds),
        "json_valid": round(1 - invalid / len(golds), 4),
        "overall": prf(total),
        "fields": {f: prf(s) for f, s in stats.items()},
    }

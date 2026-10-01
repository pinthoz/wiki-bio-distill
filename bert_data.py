"""BERT student: converts JSON labels into spans in the text (and back).

BERT does not generate text; it tags tokens. So:
  training : teacher JSON  ->  spans in the text  ->  BIO tags per token
  inference: BIO tags      ->  spans              ->  JSON (ISO dates, words as "lemma")
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict

from common import FIELDS, norm

# Span tags. Nationality and occupations can have several spans.
SPAN_FIELDS = {
    "name": "NAME",
    "birth_date": "BDATE",
    "birth_place": "BPLACE",
    "death_date": "DDATE",
    "death_place": "DPLACE",
    "nationality": "NAT",
    "occupations": "OCC",
}
TAG2FIELD = {t: f for f, t in SPAN_FIELDS.items()}
LABELS = ["O"] + [f"{p}-{t}" for t in SPAN_FIELDS.values() for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# ---------- Dates ----------
# Month names in Portuguese, as they appear in the texts
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "janeiro",
            "fevereiro",
            "março",
            "abril",
            "maio",
            "junho",
            "julho",
            "agosto",
            "setembro",
            "outubro",
            "novembro",
            "dezembro",
        ]
    )
}
DATE_RE = re.compile(
    r"(?:(?P<d>\d{1,2})(?:\.?º)?\s+de\s+)?(?P<m>"
    + "|".join(MONTHS)
    + r")\s+de\s+(?P<y>\d{3,4})"
    r"|(?P<yonly>\b\d{3,4}\b)",
    re.I,
)


def date_to_iso(text: str) -> str | None:
    """'16 de novembro de 1922' -> '1922-11-16'; 'maio de 1920' -> '1920-05'; '1109' -> '1109'."""
    m = DATE_RE.search(text)
    if not m:
        return None
    if m.group("yonly"):
        return m.group("yonly").zfill(4)
    y, mo = m.group("y").zfill(4), MONTHS[m.group("m").lower()]
    if m.group("d"):
        return f"{y}-{mo:02d}-{int(m.group('d')):02d}"
    return f"{y}-{mo:02d}"


def date_spans(text: str):
    """Every date expression in the text, with the matching ISO date."""
    return [
        (m.start(), m.end(), date_to_iso(m.group(0))) for m in DATE_RE.finditer(text)
    ]


# ---------- Words: forms in the text vs. "lemma" in the label ----------
def surface_forms(lemma: str) -> list[str]:
    """Likely forms in the text of a masculine singular (Portuguese) lemma."""
    forms = {lemma}
    rules = [
        ("ês", "esa"),
        ("or", "ora"),
        ("ão", "ã"),
        ("ão", "ona"),
        ("eu", "eia"),
        ("o", "a"),
        ("ta", "tisa"),
        ("or", "riz"),
        ("dor", "triz"),
    ]
    for a, b in rules:
        if lemma.endswith(a):
            forms.add(lemma[: -len(a)] + b)
    forms |= {f + "s" for f in list(forms)} | {f + "es" for f in list(forms)}
    return sorted(forms, key=len, reverse=True)


def _fold(s: str) -> str:
    """Lowercase without accents, keeping the length (so the indices still line up)."""
    return "".join(unicodedata.normalize("NFKD", c)[0] for c in s.lower())


def find_all(text: str, value: str):
    """Occurrences of value in the text (ignoring case and accents), as whole words."""
    t, v = _fold(text), _fold(value).strip()
    if not v:
        return []
    return [
        (m.start(), m.end())
        for m in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", t)
    ]


# ---------- JSON -> spans (for training) ----------
def align(text: str, label: dict):
    """Returns (spans, lemmas, complete). complete=False if some value was not found in the text."""
    spans, lemmas, missing, used = [], [], 0, []

    def free(s, e):
        return all(e <= a or s >= b for a, b in used)

    def add(s, e, tag):
        spans.append({"start": s, "end": e, "label": tag})
        used.append((s, e))

    # name: the first occurrence
    occ = find_all(text, label["name"]) if label.get("name") else []
    if occ:
        add(*occ[0], "NAME")
    elif label.get("name"):
        missing += 1

    # dates: the date expression whose ISO matches (birth before death)
    dates = date_spans(text)
    last = -1
    for field in ("birth_date", "death_date"):
        iso = label.get(field)
        if not iso:
            continue
        hit = next(
            ((s, e) for s, e, d in dates if d == iso and s > last and free(s, e)), None
        )
        if hit:
            add(*hit, SPAN_FIELDS[field])
            last = hit[0]
        else:
            missing += 1

    # places: birth before death (Lisboa — Lisboa: the 1st is birth, the 2nd is death)
    last = -1
    for field in ("birth_place", "death_place"):
        value = label.get(field)
        if not value:
            continue
        hit = next(
            ((s, e) for s, e in find_all(text, value) if s > last and free(s, e)), None
        )
        if hit:
            add(*hit, SPAN_FIELDS[field])
            last = hit[0]
        else:
            missing += 1

    # lists: look for each lemma in its likely feminine/plural forms
    for field in ("nationality", "occupations"):
        for lemma in label.get(field) or []:
            hit = None
            for form in surface_forms(lemma):
                hit = next(
                    ((s, e) for s, e in find_all(text, form) if free(s, e)), None
                )
                if hit:
                    break
            if hit:
                add(*hit, SPAN_FIELDS[field])
                lemmas.append((text[hit[0] : hit[1]], lemma))
            else:
                missing += 1

    return sorted(spans, key=lambda s: s["start"]), lemmas, missing == 0


def build_lemma_map(pairs) -> dict:
    """form in the text -> most frequent lemma ('portuguesa' -> 'português'), learned from the data."""
    counts = defaultdict(Counter)
    for surface, lemma in pairs:
        counts[norm(surface)][lemma] += 1
    return {s: c.most_common(1)[0][0] for s, c in counts.items()}


# ---------- spans -> JSON (at inference) ----------
def spans_from_bio(text: str, offsets, tags) -> list[dict]:
    """Merges consecutive B-/I- tokens of the same type into spans of the text."""
    spans, cur = [], None
    for (s, e), tag in zip(offsets, tags):
        if s == e:  # special tokens ([CLS], [SEP], padding)
            cur = None
            continue
        if tag == "O":
            cur = None
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "B" or cur is None or cur["label"] != kind:
            cur = {"start": s, "end": e, "label": kind}
            spans.append(cur)
        else:
            cur["end"] = e
    for sp in spans:
        sp["text"] = text[sp["start"] : sp["end"]]
    return spans


def lemmatize(surface: str, lemma_map: dict) -> str:
    key = norm(surface)
    if key in lemma_map:
        return lemma_map[key]
    word = surface.lower().strip()
    for a, b in (
        ("esas", "ês"),
        ("esa", "ês"),
        ("oras", "or"),
        ("ora", "or"),
        ("es", ""),
        ("s", ""),
    ):
        if word.endswith(a) and len(word) > len(a) + 2:
            return word[: -len(a)] + b
    return word


def spans_to_json(spans: list[dict], lemma_map: dict) -> dict:
    out = {f: None for f in FIELDS}
    out["nationality"], out["occupations"] = [], []
    for sp in spans:
        field, text = TAG2FIELD[sp["label"]], sp["text"].strip(" ,;.")
        if field in ("nationality", "occupations"):
            lemma = lemmatize(text, lemma_map)
            if lemma not in out[field]:
                out[field].append(lemma)
        elif out[field] is None:  # single-value fields: the first span wins
            out[field] = date_to_iso(text) if field.endswith("_date") else text
    if out["name"] is None:
        out["name"] = ""
    return out


def main():
    """Builds data/bert_{train,val}.jsonl and data/lemma_map.json from the train/val of step 4."""
    texts = {
        r["id"]: r["text"]
        for r in map(json.loads, open("data/raw.jsonl", encoding="utf-8"))
    }
    all_pairs, stats = [], {}
    for split in ("train", "val"):
        rows, full = [], 0
        for ex in map(json.loads, open(f"data/{split}.jsonl", encoding="utf-8")):
            label = json.loads(ex["completion"][0]["content"])
            spans, pairs, complete = align(texts[ex["id"]], label)
            full += complete
            if split == "val" or complete:  # training: only fully aligned examples
                rows.append({"id": ex["id"], "text": texts[ex["id"]], "spans": spans})
            if split == "train":
                all_pairs += pairs
        with open(f"data/bert_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = sum(1 for _ in open(f"data/{split}.jsonl", encoding="utf-8"))
        stats[split] = (full, n, len(rows))
    json.dump(
        build_lemma_map(all_pairs),
        open("data/lemma_map.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    for split, (full, n, kept) in stats.items():
        print(f"{split}: {full}/{n} fully aligned ({full / n:.0%}); kept {kept}")


if __name__ == "__main__":
    main()

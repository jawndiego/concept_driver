from __future__ import annotations

import json
import re
from typing import Any, Iterable

import pandas as pd

from concept_driver.remote_llm import RemoteLLMClient

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def slugify_query(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "llm_concepts"


def build_term_extraction_prompt(
    query: str,
    *,
    max_terms: int,
    language: str,
    concept_set_name: str,
) -> str:
    return (
        "Return only valid JSON. No markdown, no prose, no explanation.\n"
        f"Build one concept set named {concept_set_name!r} for the query {query!r}.\n"
        f"Return up to {max_terms} distinct related words or short noun phrases.\n"
        "Prefer concrete, semantically related concepts, not sentences.\n"
        f"Use {language!r} as the language tag for every term unless the query clearly requires another tag.\n"
        "JSON schema:\n"
        '{\n'
        '  "terms": [\n'
        '    {"term": "hero", "label": "hero", "language": "en"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- each term should be 1 to 4 words\n"
        "- no duplicates\n"
        "- no numbering\n"
        "- no commentary\n"
    )


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    if stripped:
        yield stripped

    for match in JSON_FENCE_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1].strip()
            if candidate:
                yield candidate


def _extract_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("terms", "concepts", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if any(key in payload for key in ("term", "label", "word", "concept")):
            return [payload]
    raise ValueError("LLM response did not contain a JSON list of terms.")


def parse_llm_terms_response(
    text: str,
    *,
    concept_set_name: str,
    default_language: str,
    max_terms: int,
) -> pd.DataFrame:
    payload: Any | None = None
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        raise ValueError("LLM response did not contain valid JSON.")

    items = _extract_items(payload)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if len(records) >= max_terms:
            break

        if isinstance(item, str):
            term = item.strip()
            label = term
            language = default_language
        elif isinstance(item, dict):
            raw_term = item.get("term") or item.get("label") or item.get("word") or item.get("concept")
            term = str(raw_term).strip() if raw_term is not None else ""
            if not term:
                continue
            raw_label = item.get("label")
            label = str(raw_label).strip() if raw_label is not None else term
            raw_language = item.get("language")
            language = str(raw_language).strip() if raw_language is not None else default_language
        else:
            continue

        if not term:
            continue

        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "concept_set": concept_set_name,
                "term": term,
                "label": label or term,
                "language": language or default_language,
                "order": len(records) + 1,
            }
        )

    if not records:
        raise ValueError("LLM response did not yield any usable terms.")

    return pd.DataFrame.from_records(records)


def generate_concepts_from_llm(
    client: RemoteLLMClient,
    *,
    query: str,
    concept_set_name: str | None = None,
    language: str = "en",
    max_terms: int = 24,
) -> pd.DataFrame:
    resolved_set_name = concept_set_name or slugify_query(query)
    prompt = build_term_extraction_prompt(
        query,
        max_terms=max_terms,
        language=language,
        concept_set_name=resolved_set_name,
    )
    response = client.ask(
        prompt,
        max_tokens=max(256, max_terms * 40),
        temperature=0.2,
    )
    return parse_llm_terms_response(
        response,
        concept_set_name=resolved_set_name,
        default_language=language,
        max_terms=max_terms,
    )

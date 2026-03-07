from concept_driver.llm_concepts import parse_llm_terms_response, slugify_query


def test_parse_llm_terms_response_accepts_json_dict() -> None:
    response = """
    {
      "terms": [
        {"term": "hero", "label": "hero", "language": "en"},
        {"term": "mentor", "label": "mentor", "language": "en"},
        {"term": "Hero", "label": "Hero", "language": "en"}
      ]
    }
    """

    df = parse_llm_terms_response(
        response,
        concept_set_name="hero",
        default_language="en",
        max_terms=10,
    )

    assert list(df["term"]) == ["hero", "mentor"]
    assert list(df["concept_set"].unique()) == ["hero"]


def test_parse_llm_terms_response_accepts_json_fence_and_string_items() -> None:
    response = """```json
    ["grief", "mourning", "loss"]
    ```"""

    df = parse_llm_terms_response(
        response,
        concept_set_name="grief",
        default_language="en",
        max_terms=2,
    )

    assert list(df["term"]) == ["grief", "mourning"]
    assert list(df["language"]) == ["en", "en"]


def test_slugify_query_normalizes_text() -> None:
    assert slugify_query("Hero Archetypes!") == "hero_archetypes"

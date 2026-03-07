from concept_driver.data import build_concepts_from_corpus


def test_build_concepts_from_corpus_extracts_real_terms() -> None:
    corpus = "Hero hero village village village river river lantern."
    concepts = build_concepts_from_corpus(corpus, min_freq=2, max_terms=10)

    assert not concepts.empty
    assert list(concepts["term"])[:3] == ["village", "hero", "river"]

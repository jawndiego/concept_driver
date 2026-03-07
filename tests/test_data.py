from concept_driver.data import extract_contexts, sentence_split


def test_sentence_split_normalizes_whitespace() -> None:
    text = "January is cold.   February is snowy!\nMarch arrives?"
    assert sentence_split(text) == [
        "January is cold.",
        "February is snowy!",
        "March arrives?",
    ]


def test_extract_contexts_uses_sentence_window() -> None:
    corpus = "January is cold. February is snowy. March is windy."
    contexts = extract_contexts(corpus, ["February"], max_contexts=5, context_window=1)
    assert contexts["February"] == [
        "January is cold. February is snowy. March is windy."
    ]

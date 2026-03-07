from pathlib import Path

from concept_driver.query import build_query_session


def test_query_session_returns_neighbors_and_contexts(tmp_path: Path) -> None:
    concepts_path = tmp_path / "concepts.csv"
    corpus_path = tmp_path / "corpus.txt"

    concepts_path.write_text(
        "\n".join(
            [
                "concept_set,term,label,language,order",
                "months,January,January,en,1",
                "months,July,July,en,7",
                "colors,red,red,en,1",
                "colors,blue,blue,en,2",
            ]
        ),
        encoding="utf-8",
    )
    corpus_path.write_text(
        "January is cold. July is warm. Red suggests warmth. Blue suggests sea.",
        encoding="utf-8",
    )

    session = build_query_session(
        concepts_path=str(concepts_path),
        corpus_path=str(corpus_path),
        mode="context",
        encoder="tfidf",
    )
    result = session.query("January", top_k=2)

    assert result.known_term is True
    assert result.concept_sets == ["months"]
    assert result.contexts
    assert len(result.neighbors) == 2
    assert all(match.term != "January" for match in result.neighbors)

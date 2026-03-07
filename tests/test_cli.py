from pathlib import Path

import pandas as pd

from concept_driver.cli import build_reports, main, parse_args


def test_build_reports_writes_manifest_and_index(tmp_path: Path) -> None:
    concepts_path = tmp_path / "concepts.csv"
    corpus_path = tmp_path / "corpus.txt"
    output_dir = tmp_path / "report"

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

    args = parse_args(
        [
            "report",
            "--concepts",
            str(concepts_path),
            "--corpus",
            str(corpus_path),
            "--out",
            str(output_dir),
            "--mode",
            "context",
            "--encoder",
            "tfidf",
        ]
    )

    written_dir = build_reports(args)

    assert written_dir == output_dir
    assert (output_dir / "index.html").exists()
    assert (output_dir / "manifest.csv").exists()
    assert (output_dir / "resolved_concepts.csv").exists()
    assert (output_dir / "months.html").exists()
    assert (output_dir / "colors.html").exists()

    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert set(manifest["concept_set"]) == {"months", "colors"}


def test_sample_command_writes_example_report(tmp_path: Path) -> None:
    exit_code = main(["sample", "--out", str(tmp_path / "sample-report")])
    assert exit_code == 0
    assert (tmp_path / "sample-report" / "index.html").exists()


def test_tui_without_real_data_source_fails() -> None:
    exit_code = main(["tui"])
    assert exit_code == 1


def test_tui_accepts_remote_llm_without_local_source(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError()))
    exit_code = main(
        [
            "tui",
            "--llm-base-url",
            "https://example.com/v1",
        ]
    )
    assert exit_code == 0


def test_build_reports_from_llm_query(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "llm-report"

    class StubClient:
        def __init__(self) -> None:
            self.model = None

    def fake_build_llm_client(_args):
        return StubClient()

    def fake_generate_concepts_from_llm(client, *, query, concept_set_name, language, max_terms):
        assert query == "hero archetypes"
        assert concept_set_name is None
        assert language == "en"
        assert max_terms == 24
        client.model = "resolved-model"
        return pd.DataFrame(
            {
                "concept_set": ["hero_archetypes", "hero_archetypes", "hero_archetypes"],
                "term": ["hero", "mentor", "shadow"],
                "label": ["hero", "mentor", "shadow"],
                "language": ["en", "en", "en"],
                "order": [1, 2, 3],
            }
        )

    monkeypatch.setattr("concept_driver.cli.build_llm_client", fake_build_llm_client)
    monkeypatch.setattr("concept_driver.cli.generate_concepts_from_llm", fake_generate_concepts_from_llm)

    args = parse_args(
        [
            "report",
            "--out",
            str(output_dir),
            "--mode",
            "term",
            "--encoder",
            "tfidf",
            "--llm-base-url",
            "https://example.com/v1",
            "--llm-query",
            "hero archetypes",
        ]
    )

    written_dir = build_reports(args)

    assert written_dir == output_dir
    assert (output_dir / "index.html").exists()
    assert (output_dir / "manifest.csv").exists()
    assert (output_dir / "resolved_concepts.csv").exists()

    resolved = pd.read_csv(output_dir / "resolved_concepts.csv")
    assert list(resolved["term"]) == ["hero", "mentor", "shadow"]
    assert args.llm_model == "resolved-model"

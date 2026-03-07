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
            "--llm-model",
            "test-model",
        ]
    )
    assert exit_code == 0

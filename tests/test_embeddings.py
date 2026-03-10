from types import SimpleNamespace

import numpy as np

from concept_driver.embeddings import GeminiBackend, resolve_model_name


def test_resolve_model_name_uses_gemini_default() -> None:
    assert resolve_model_name("gemini", None) == "gemini-embedding-001"
    assert resolve_model_name("gemini", "all-MiniLM-L6-v2") == "gemini-embedding-001"
    assert resolve_model_name("gemini", "gemini-embedding-001") == "gemini-embedding-001"


def test_gemini_backend_requests_embeddings(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModels:
        def embed_content(self, *, model, contents, config=None):
            calls.append(
                {
                    "model": model,
                    "contents": list(contents),
                    "config": getattr(config, "kwargs", None),
                }
            )
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[3.0, 4.0]),
                    SimpleNamespace(values=[0.0, 5.0]),
                ]
            )

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(
        "concept_driver.embeddings.genai",
        SimpleNamespace(Client=FakeClient),
    )
    monkeypatch.setattr(
        "concept_driver.embeddings.genai_types",
        SimpleNamespace(EmbedContentConfig=FakeConfig),
    )

    backend = GeminiBackend(
        model_name="gemini-embedding-001",
        api_key="test-key",
        task_type="CLUSTERING",
        output_dimensionality=768,
    )
    matrix = backend.fit_transform(["hero", "mentor"])

    assert matrix.shape == (2, 2)
    assert np.allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
    assert calls == [
        {
            "model": "gemini-embedding-001",
            "contents": ["hero", "mentor"],
            "config": {
                "task_type": "CLUSTERING",
                "output_dimensionality": 768,
            },
        }
    ]

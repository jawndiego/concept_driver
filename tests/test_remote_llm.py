from concept_driver.remote_llm import RemoteLLMClient, extract_chat_text


def test_extract_chat_text_from_string_content() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "hello world"
                }
            }
        ]
    }
    assert extract_chat_text(payload) == "hello world"


def test_remote_llm_client_normalizes_base_url() -> None:
    client = RemoteLLMClient(
        base_url="https://example.com/v1/",
        model="test-model",
    )
    assert client.base_url == "https://example.com/v1"


def test_remote_llm_client_resolves_model_from_list(monkeypatch) -> None:
    client = RemoteLLMClient(base_url="https://example.com/v1")
    monkeypatch.setattr(client, "list_models", lambda: ["resolved-model"])

    assert client.resolve_model() == "resolved-model"
    assert client.model == "resolved-model"

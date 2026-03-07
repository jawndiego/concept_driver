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

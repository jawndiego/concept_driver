import pytest
from fastapi import HTTPException

from concept_driver.railway_api import prepare_chat_payload


def test_prepare_chat_payload_sets_default_model() -> None:
    payload = prepare_chat_payload(
        {"messages": [{"role": "user", "content": "hi"}]},
        default_model="Qwen/Qwen3.5-397B-A17B",
    )
    assert payload["model"] == "Qwen/Qwen3.5-397B-A17B"


def test_prepare_chat_payload_rejects_streaming() -> None:
    with pytest.raises(HTTPException) as excinfo:
        prepare_chat_payload(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            default_model="Qwen/Qwen3.5-397B-A17B",
        )
    assert excinfo.value.status_code == 501

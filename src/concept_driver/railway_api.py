from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class Settings:
    hf_token: str
    hf_model: str
    hf_base_url: str
    proxy_api_key: str | None
    request_timeout_seconds: float


def load_settings() -> Settings:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="HF_TOKEN is required.")

    return Settings(
        hf_token=hf_token,
        hf_model=os.getenv("HF_MODEL", DEFAULT_MODEL),
        hf_base_url=os.getenv("HF_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        proxy_api_key=os.getenv("PROXY_API_KEY"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300")),
    )


app = FastAPI(title="Concept Driver Qwen Proxy", version="0.1.0")


@lru_cache
def get_settings() -> Settings:
    return load_settings()


class SimpleChatRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_tokens: int | None = Field(default=512, ge=1)
    temperature: float | None = Field(default=0.7, ge=0.0)


def prepare_chat_payload(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    forwarded = dict(payload)
    forwarded.setdefault("model", default_model)
    if forwarded.get("stream"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming passthrough is not implemented in this proxy.",
        )
    if "messages" not in forwarded:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="messages is required",
        )
    return forwarded


def require_proxy_auth(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.proxy_api_key:
        return

    expected = f"Bearer {settings.proxy_api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


async def forward_chat(payload: dict[str, Any]) -> JSONResponse:
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.hf_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    return JSONResponse(status_code=response.status_code, content=response.json())


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ok": True,
        "model": settings.hf_model,
        "upstream": settings.hf_base_url,
        "proxy_auth_enabled": settings.proxy_api_key is not None,
    }


@app.get("/v1/models", dependencies=[Depends(require_proxy_auth)])
async def models() -> dict[str, Any]:
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.hf_model,
                "object": "model",
                "owned_by": "huggingface-router",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_proxy_auth)])
async def chat_completions(request: Request) -> JSONResponse:
    settings = get_settings()
    payload = await request.json()
    forwarded = prepare_chat_payload(payload, default_model=settings.hf_model)
    return await forward_chat(forwarded)


@app.post("/chat", dependencies=[Depends(require_proxy_auth)])
async def simple_chat(body: SimpleChatRequest) -> JSONResponse:
    settings = get_settings()
    messages: list[dict[str, str]] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append({"role": "user", "content": body.prompt})

    payload = {
        "model": settings.hf_model,
        "messages": messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }
    return await forward_chat(payload)

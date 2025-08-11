"""Runtime configuration with strict validation.

This keeps misconfiguration errors out of runtime execution.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

EngineType = Literal["chat", "query", "retriever"]
StreamFmt = Literal["text", "jsonl"]

class Params(BaseModel):
    adapter: str = "llama_index"                # e.g., "llama_index", "langchain", "crewai"
    engine_type: EngineType = "chat"

    resolver: str = "uc_alias"                  # "uc_alias" | "static_uri" | custom
    model_ref: str                               # e.g., models:/cat.sch.model/Production OR runs:/... OR pkg.module:factory

    # common knobs
    top_k: int = Field(default=4, ge=1, le=100)

    # memory (optional pluggable)
    memory_backend: str = "inproc"              # "inproc" | "redis" | "uc_delta"
    redis_url: Optional[str] = None
    uc_delta_table: Optional[str] = None

    # runtime
    timeout_s: int = Field(default=120, ge=1, le=900)
    stream_format: StreamFmt = "text"

    # hot reload (if resolver supports it)
    hot_reload: bool = True
    hot_reload_interval_s: int = Field(default=30, ge=5, le=600)

    # reliability
    max_concurrent_streams: int = Field(default=64, ge=1, le=10000)
    drain_seconds: int = Field(default=10, ge=1, le=300)

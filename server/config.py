from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

Framework = Literal["llamaindex", "langchain"]
EngineType = Literal["chat", "query", "retriever"]
StreamFmt = Literal["text", "jsonl"]

class Params(BaseModel):
    framework: Framework = "llamaindex"

    # LlamaIndex (Unity Catalog stage URI)
    uc_model_uri: Optional[str] = None                 # models:/catalog.schema.name/Production
    engine_type: EngineType = "chat"
    top_k: int = Field(default=4, ge=1, le=100)

    # LangChain
    langchain_factory: Optional[str] = None            # "pkg.mod:build_chain"
    use_history: bool = True

    # Runtime & streaming
    timeout_s: int = Field(default=120, ge=1, le=900)
    stream_format: StreamFmt = "text"

    # Hot reload (UC)
    hot_reload: bool = True
    hot_reload_interval_s: int = Field(default=30, ge=5, le=600)

    # Reliability
    max_concurrent_streams: int = Field(default=64, ge=1, le=10000)
    drain_seconds: int = Field(default=10, ge=1, le=300)

    @validator("uc_model_uri", always=True)
    def _need_uc_for_llama(cls, v, values):
        if values.get("framework") == "llamaindex" and not v:
            raise ValueError("uc_model_uri is required for framework=llamaindex")
        return v

    @validator("langchain_factory", always=True)
    def _need_factory_for_lc(cls, v, values):
        if values.get("framework") == "langchain" and not v:
            raise ValueError("langchain_factory is required for framework=langchain")
        return v

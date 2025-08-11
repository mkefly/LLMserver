"""I/O helpers for MLServer requests and streaming chunks."""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Tuple
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import decode
from mlserver.codecs.string import StringRequestCodec, StringCodec

async def decode_input(req: InferenceRequest) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Decodes input as raw text or JSON with {input, session_id, params}."""
    if not getattr(req, "inputs", None):
        raise ValueError("No inputs provided")
    try:
        payload = await decode(req, default_codec=StringRequestCodec)
        if isinstance(payload, str):
            return payload, None, {}
        if isinstance(payload, dict):
            txt = payload.get("input")
            if not isinstance(txt, str):
                raise ValueError("JSON payload must contain string 'input'")
            return txt, payload.get("session_id"), (payload.get("params") or {})
    except Exception:
        pass
    first = req.inputs[0]
    txt = StringCodec.decode_input(first)  # type: ignore[attr-defined]
    return txt, None, {}

def pack_text(name: str, text: str, model: str) -> InferenceResponse:
    """Packs a single BYTES output tensor for MLServer."""
    return InferenceResponse(model_name=model, outputs=[
        ResponseOutput(name=name, shape=[1], datatype="BYTES", data=[text])
    ])

def pack_chunk(model: str, token: str, fmt: str) -> InferenceResponse:
    """Packs a streaming token in text or JSONL."""
    if fmt == "jsonl":
        token = json.dumps({"type": "token", "data": token}, separators=(",", ":"))
    return pack_text("token", token, model)

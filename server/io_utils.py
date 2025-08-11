import json
from typing import Any, Dict, Optional, Tuple
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import decode
from mlserver.codecs.string import StringRequestCodec, StringCodec

async def decode_input(req: InferenceRequest) -> Tuple[str, Optional[str], Dict[str, Any]]:
    if not req.inputs:
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
    txt = StringCodec.decode_input(first)
    return txt, None, {}

def pack_text(name: str, text: str, model: str) -> InferenceResponse:
    return InferenceResponse(
        model_name=model,
        outputs=[ResponseOutput(name=name, shape=[1], datatype="BYTES", data=[text])]
    )

def pack_chunk(model: str, token: str, fmt: str) -> InferenceResponse:
    if fmt == "jsonl":
        token = json.dumps({"type": "token", "data": token}, separators=(",", ":"))
    return pack_text("token", token, model)

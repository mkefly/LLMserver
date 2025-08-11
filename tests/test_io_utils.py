import pytest
from types import SimpleNamespace
from server.io_utils import decode_input, pack_text, pack_chunk

class FakeInput:
    def __init__(self, data: bytes):
        self.data = data
        self.datatype = "BYTES"
        self.shape = [1]
        class Contents:
            def __init__(self, b): self.bytes_contents = [b]
        self.contents = Contents(data)

@pytest.mark.asyncio
async def test_decode_input_dict_payload(monkeypatch):
    from server import io_utils as iu
    async def fake_decode(req, default_codec=None):
        return {"input": "hi", "session_id": "s1", "params": {"k": 1}}
    monkeypatch.setattr(iu, "decode", fake_decode)

    req = SimpleNamespace(inputs=[FakeInput(b"{}")])
    text, sid, params = await decode_input(req)
    assert text == "hi" and sid == "s1" and params == {"k": 1}

@pytest.mark.asyncio
async def test_decode_input_str_payload(monkeypatch):
    from server import io_utils as iu
    async def fake_decode(req, default_codec=None):
        return "hello"
    monkeypatch.setattr(iu, "decode", fake_decode)
    req = SimpleNamespace(inputs=[FakeInput(b"hello")])
    text, sid, params = await decode_input(req)
    assert text == "hello" and sid is None and params == {}

@pytest.mark.asyncio
async def test_decode_input_fallback_to_first_tensor(monkeypatch):
    from server import io_utils as iu
    async def fake_decode(req, default_codec=None):
        raise RuntimeError("fallback")
    monkeypatch.setattr(iu, "decode", fake_decode)
    req = SimpleNamespace(inputs=[FakeInput(b"raw")])
    text, sid, params = await decode_input(req)
    assert text == "raw"

@pytest.mark.asyncio
async def test_decode_input_no_inputs_raises():
    req = SimpleNamespace(inputs=[])
    with pytest.raises(ValueError):
        await decode_input(req)

def test_pack_text_and_chunk_jsonl():
    r1 = pack_text("out", "ok", "m")
    assert r1.outputs[0].data == ["ok"]
    r2 = pack_chunk("m", "tok", "jsonl")
    assert r2.outputs[0].data[0].startswith('{"type":"token","data":"tok"}')

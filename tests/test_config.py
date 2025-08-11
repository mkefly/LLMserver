import pytest
from server.config import Params

def test_llama_requires_uc_uri():
    with pytest.raises(Exception):
        Params(framework="llamaindex")  # missing uc_model_uri

def test_langchain_requires_factory():
    with pytest.raises(Exception):
        Params(framework="langchain")

def test_valid_llamaindex():
    p = Params(framework="llamaindex", uc_model_uri="models:/main.default.m/Production")
    assert p.engine_type == "chat"
    assert p.timeout_s == 120

def test_stream_format_validation():
    with pytest.raises(Exception):
        Params(framework="llamaindex", uc_model_uri="models:/c.s.m/Production", stream_format="bad")

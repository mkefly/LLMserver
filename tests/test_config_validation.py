import pytest
from server.config import Params

def test_engine_type_validation():
    with pytest.raises(Exception):
        Params(framework="llamaindex", uc_model_uri="models:/c.s.m/P", engine_type="bad")  # type: ignore

def test_llama_valid_defaults():
    p = Params(framework="llamaindex", uc_model_uri="models:/c.s.m/Production")
    assert p.engine_type == "chat"
    assert p.stream_format == "text"
    assert p.hot_reload

def test_langchain_valid():
    p = Params(framework="langchain", langchain_factory="pkg.mod:build")
    assert p.use_history is True

def test_timeout_bounds():
    with pytest.raises(Exception):
        Params(framework="langchain", langchain_factory="a:b", timeout_s=0)

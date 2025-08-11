import pytest
from server.config import Params

def test_need_uc_table_for_uc_delta():
    with pytest.raises(Exception):
        Params(adapter="llama_index", resolver="static_uri", model_ref="models:/x", memory_backend="uc_delta")

def test_valid_defaults_llamaindex():
    p = Params(adapter="llama_index", resolver="uc_stage", model_ref="models:/c.s.m/Production")
    assert p.engine_type == "chat"
    assert p.timeout_s == 120

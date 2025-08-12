from ..registry import ADAPTERS
from .base_mlflow_adapter import BaseMLflowAdapter

@ADAPTERS.register("crewai")
def crewai_factory(model_uri: str, **_: dict) -> BaseMLflowAdapter:
    return CrewAIAdapter(model_uri)

class CrewAIAdapter(BaseMLflowAdapter):
    FRAMEWORK_NAME = "crewai"
    SUPPORTED_MODES = ("chat", "retrieve")
    METHOD_MAP = {
        "chat": (("run",), ("stream",)),
        "retrieve": (("run",), ("stream",)),
    }

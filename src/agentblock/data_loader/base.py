import abc
from typing import Dict, Any, List
from langchain.schema import Document
from agentblock.data_loader.loader_registry import LOADER_IMPL_MAP


class AbstractDataLoaderNode(abc.ABC):
    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: str,
        config: Dict[str, Any] = None,
    ):
        self.name = name
        self.input_keys = input_keys
        self.output_key = output_key
        self.config = config or {}

    @abc.abstractmethod
    def load_data(self, inputs: Dict[str, Any]) -> List[Document]:
        pass

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        loader_inputs = {key: state.get(key) for key in self.input_keys}
        documents = self.load_data(loader_inputs)
        state[self.output_key] = documents
        return state


class GenericLoaderNode(AbstractDataLoaderNode):
    """
    단일 Node 클래스. config.loader_kind 로 구분된 다양한 로더 함수를 내부적으로 사용.
    + config.args, config.kwargs 형태로 'args'와 'kwargs'도 받아서 로더 함수에 전달
    """

    def load_data(self, inputs: Dict[str, Any]) -> List[Document]:
        loader_kind = self.config.get("loader_kind")
        if not loader_kind:
            raise ValueError("config.loader_kind is required")

        # 로더 함수 찾기
        loader_func = LOADER_IMPL_MAP.get(loader_kind)
        if not loader_func:
            raise ValueError(f"Unsupported loader_kind: {loader_kind}")

        # config에서 args/kwargs 추출 (없으면 기본 값)
        args = self.config.get("args", [])
        kwargs = self.config.get("kwargs", {})

        # 로더 함수에 inputs, args, kwargs를 전달
        # => loader_func(inputs, *args, **kwargs)
        return loader_func(inputs, *args, **kwargs)

    @staticmethod
    def from_yaml(node_config: dict, *args, **kwargs) -> "GenericLoaderNode":
        return GenericLoaderNode(
            name=node_config["name"],
            input_keys=node_config.get("input_keys", []),
            output_key=node_config["output_key"],
            config=node_config.get("config", {}),
        )

    def build(self):
        def node_fn(state: dict):
            return self.invoke(state)

        return node_fn

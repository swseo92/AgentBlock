from typing import Dict, Any


from abc import ABC, abstractmethod


class BaseComponent(ABC):
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    @abstractmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, Any]
    ) -> "BaseComponent":
        pass

    @abstractmethod
    def build(self):
        pass


class BaseNode(BaseComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_keys = []
        self.output_key = None

    @staticmethod
    @abstractmethod
    def from_yaml(config: dict, base_dir, references_map: Dict[str, Any]) -> "BaseNode":
        pass

    @abstractmethod
    def build(self) -> callable:
        pass

    def get_inputs(self, state):
        state_dict = dict(state)

        inputs = dict()
        for k in self.input_keys:
            src_key, dest_key = self.parse_input_keys(k)
            # 내부적으로 반환된 internal_key를 사용하여 inputs에 값을 추가
            inputs[dest_key] = state_dict[src_key]
        return inputs

    @staticmethod
    def parse_input_keys(input_key):
        # 매핑이 있는 경우
        if "->" in input_key:
            src_key, dest_key = input_key.split("->")
            src_key = src_key.strip()
            dest_key = dest_key.strip()
        # 매핑이 없는 경우
        else:
            src_key, dest_key = input_key.strip(), input_key.strip()  # 그대로 반환
        return src_key, dest_key


class BaseReference(BaseComponent):
    """
    비실행 노드(embedding, vector_store, loader 등)
    BFS와 무관하지만, from_yaml/build 패턴은 Node와 유사.
    """

    @staticmethod
    @abstractmethod
    def from_yaml(
        config: dict, base_dir, references_map: Dict[str, Any]
    ) -> "BaseReference":
        pass

    @abstractmethod
    def build(self) -> None:
        pass

from typing import Dict


from abc import ABC, abstractmethod


class BaseComponent(ABC):
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    @abstractmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, "BaseReference"]
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
    def from_yaml(
        config: dict, base_dir, references_map: Dict[str, "BaseReference"]
    ) -> "BaseNode":
        pass

    @abstractmethod
    def build(self) -> callable:
        pass

    def get_inputs(self, state):
        state_dict = dict(state)
        inputs = {k: state_dict[k] for k in self.input_keys if k in state_dict}
        return inputs


class BaseReference(BaseComponent):
    """
    비실행 노드(embedding, vector_store, loader 등)
    BFS와 무관하지만, from_yaml/build 패턴은 Node와 유사.
    """

    @staticmethod
    @abstractmethod
    def from_yaml(
        config: dict, base_dir, references_map: Dict[str, "BaseReference"]
    ) -> "BaseReference":
        pass

    @abstractmethod
    def build(self) -> None:
        pass

from abc import ABC, abstractmethod


class BaseNode(ABC):
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    @abstractmethod
    def from_yaml(config: dict) -> "BaseNode":
        pass

    @abstractmethod
    def build(self) -> callable:
        pass

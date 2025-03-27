from abc import ABC, abstractmethod
from agentblock.tools.load_config import (
    get_yaml_for_single_node_file,
    get_parent_dir_abspath,
)


class BaseNode(ABC):
    def __init__(self, name: str):
        self.name = name
        self.input_keys = []
        self.output_key = None

    @staticmethod
    @abstractmethod
    def from_yaml(config: dict) -> "BaseNode":
        pass

    @abstractmethod
    def build(self) -> callable:
        pass

    def get_inputs(self, state):
        state_dict = dict(state)
        inputs = {k: state_dict[k] for k in self.input_keys if k in state_dict}
        return inputs

    def from_yaml_file_single_node(self, yaml_path):
        config = get_yaml_for_single_node_file(yaml_path)
        base_dir = get_parent_dir_abspath(yaml_path)
        return self.from_yaml(config, base_dir)

from agentblock.base import BaseNode
from functions import merge


class FunctionNode(BaseNode):
    def __init__(self, name, function_name, input_keys, output_key):
        super().__init__(name)
        self.function_name = function_name
        self.input_keys = input_keys
        self.output_key = output_key

    @staticmethod
    def from_yaml(config: dict) -> "FunctionNode":
        return FunctionNode(
            name=config["name"],
            function_name=config["function"],
            input_keys=config["input_keys"],
            output_key=config["output_key"],
        )

    def build(self):
        fn = getattr(merge, self.function_name)

        def wrapped(state):
            return fn(state)

        return wrapped

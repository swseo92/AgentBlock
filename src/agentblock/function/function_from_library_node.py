from typing import Any, Dict, List, Union
import importlib
import importlib.util

from agentblock.function.base import FunctionNode


class FunctionFromLibraryNode(FunctionNode):
    """
    'from_library': "module:function" 형식
    raw_result를 dict가 아닐 수도 있으니, 상위 _wrap_result() 로직으로 감쌀 것.
    """

    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: Union[str, List[str]],
        base_dir: str = None,
        from_library: str = None,
        params: List[str] = None,
    ):
        super().__init__(name, input_keys, output_key)
        self.base_dir = base_dir
        self.from_library = from_library
        self.params = params or []
        self._func = None

    @staticmethod
    def from_yaml(config: dict, base_dir: str = None) -> "FunctionFromLibraryNode":
        if "output_key" not in config:
            raise ValueError(f"Missing 'output_key' in config: {config.get('name')}")

        from_lib = config["config"].get("from_library")
        if not from_lib:
            raise ValueError("Must provide 'from_library' in config")

        return FunctionFromLibraryNode(
            name=config["name"],
            input_keys=config.get("input_keys", []),
            output_key=config["output_key"],
            base_dir=base_dir,
            from_library=from_lib,
            params=config["config"].get("params", []),
        )

    def parse_config(self, config: dict = None, base_dir: str = None):
        """이미 __init__에서 처리했으므로 pass."""
        pass

    def import_target_function(self):
        if not self.from_library or ":" not in self.from_library:
            raise ValueError(f"Invalid from_library: {self.from_library}")

        mod_part, func_name = self.from_library.split(":", 1)

        module = importlib.import_module(mod_part)
        target_func = getattr(module, func_name)

        if not callable(target_func):
            raise TypeError(f"'{func_name}' is not callable in {mod_part}")

        self._func = target_func

    def call_target_function(self, inputs: Dict[str, Any]) -> Any:
        # input_keys + params(list[str]) => call _func
        extra_args = {}
        for p in self.params:
            if p in inputs:
                extra_args[p] = inputs[p]

        # remove them from inputs if they overlap
        for p in self.params:
            inputs.pop(p, None)

        # now call
        raw_result = self._func(**inputs, **extra_args)
        # raw_result could be anything (int/str/list/dict)

        return raw_result

import os
import importlib
import importlib.util
from typing import Any, Dict, List, Union

from agentblock.function.base import FunctionNode
from agentblock.tools.load_config import get_abspath


class FunctionFromFileNode(FunctionNode):
    """
    기존의 'function_path' 로직을 구현 (구 FunctionNode).
    - parse_config: config에서 function_path, params 가져옴
    - import_target_function: importlib 통해 파일 로드
    - call_target_function: dict 반환 함수 실행
    """

    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: Union[str, List[str]],
        base_dir: str = None,
        function_path: str = None,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name, input_keys, output_key)
        self.base_dir = base_dir
        self.function_path = function_path
        self.params = params or {}

        self._loaded_func = None

    @staticmethod
    def from_yaml(config: dict, base_dir: str = None) -> "FunctionFromFileNode":
        if "output_key" not in config:
            raise ValueError(f"Missing 'output_key' in config: {config.get('name')}")

        return FunctionFromFileNode(
            name=config["name"],
            input_keys=config.get("input_keys", []),
            output_key=config["output_key"],
            base_dir=base_dir,
            function_path=config["config"]["function_path"],
            params=config["config"].get("params", {}),
        )

    def parse_config(self, config: dict = None, base_dir: str = None):
        """우리는 이미 __init__에서 가져왔으므로 여기선 pass."""
        pass

    def import_target_function(self):
        if not self.function_path:
            raise ValueError("No function_path specified for FunctionFromFileNode")

        # parse "module.py:function_name"
        if ":" not in self.function_path:
            raise ValueError(
                f"Invalid function_path '{self.function_path}'. Must be 'xxx:func'"
            )

        mod_part, func_name = self.function_path.rsplit(":", 1)
        mod_part = get_abspath(mod_part, self.base_dir).replace(".", "/")
        mod_part_fs = mod_part + ".py"
        mod_file = os.path.abspath(mod_part_fs)

        if not os.path.isfile(mod_file):
            raise FileNotFoundError(f"Cannot find file: {mod_file}")

        spec = importlib.util.spec_from_file_location("temp_loaded_module", mod_file)
        if not spec or not spec.loader:
            raise ImportError(f"Failed to create spec for '{mod_file}'")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, func_name):
            raise AttributeError(f"Module '{mod_file}' has no attribute '{func_name}'")
        func = getattr(module, func_name)

        if not callable(func):
            raise TypeError(f"Attribute '{func_name}' is not callable in '{mod_file}'")

        self._loaded_func = func

    def call_target_function(self, inputs: Dict[str, Any]) -> Any:
        # inputs + self.params => dict
        final_inputs = dict(inputs)
        final_inputs.update(self.params)

        result = self._loaded_func(**final_inputs)
        # if not isinstance(result, dict):
        #     raise ValueError(
        #         f"Function '{self.function_path}' must return a dict, got {type(result).__name__}"
        #     )
        # # 여기서 raw_result는 dict이지만, 상위클래스에서 _wrap_result()로 또 감쌀 예정
        # # => 최종구조: {self.output_key: { ... }} 이 될 수도 있음
        return result

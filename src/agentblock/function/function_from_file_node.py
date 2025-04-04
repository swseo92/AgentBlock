import os
import importlib
import importlib.util
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass

from agentblock.function.base import FunctionNode, FunctionResult
from agentblock.tools.load_config import get_abspath


class FunctionFromFileNode(FunctionNode[Dict[str, Any]]):
    """
    기존의 'function_path' 로직을 구현 (구 FunctionNode).
    - parse_config: config에서 function_path, param 가져옴
    - import_target_function: importlib 통해 파일 로드
    - call_target_function: dict 반환 함수 실행
    """

    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: Union[str, List[str]],
        base_dir: Optional[str] = None,
        function_path: Optional[str] = None,
        param: Optional[Dict[str, Any]] = None,
        validate_inputs: bool = True,
        validate_outputs: bool = True,
    ):
        super().__init__(
            name=name,
            input_keys=input_keys,
            output_key=output_key,
            validate_inputs=validate_inputs,
            validate_outputs=validate_outputs,
        )
        self.base_dir = base_dir
        self.function_path = function_path
        self.param = param or {}
        self._loaded_func = None

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, Any]
    ) -> "FunctionFromFileNode":
        if "output_key" not in config:
            raise ValueError(f"Missing 'output_key' in config: {config.get('name')}")

        return FunctionFromFileNode(
            name=config["name"],
            input_keys=config.get("input_keys", []),
            output_key=config["output_key"],
            base_dir=base_dir,
            function_path=config["config"]["function_path"],
            param=config["config"].get("param", {}),
            validate_inputs=config["config"].get("validate_inputs", True),
            validate_outputs=config["config"].get("validate_outputs", True),
        )

    def parse_config(self, config: dict = None, base_dir: Optional[str] = None) -> None:
        """우리는 이미 __init__에서 가져왔으므로 여기선 pass."""
        pass

    def import_target_function(self) -> None:
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

    def call_target_function(self, inputs: Dict[str, Any]) -> FunctionResult[Dict[str, Any]]:
        # inputs + self.param => dict
        final_inputs = dict(inputs)
        final_inputs.update(self.param)

        result = self._loaded_func(**final_inputs)
        
        # 메타데이터 추가
        metadata = {
            "function_path": self.function_path,
            "param": self.param,
        }
        
        return FunctionResult(value=result, metadata=metadata)

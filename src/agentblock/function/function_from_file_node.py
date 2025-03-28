import os
import importlib
import importlib.util
from typing import Any, Dict, List, Union

from agentblock.base import BaseNode
from agentblock.tools.load_config import get_abspath


def load_function_from_path(function_path: str, base_dir: str | None = None):
    """
    function_path 예) "../test_funcs.test_funcs_for_validation:partial_values_func"
    base_dir: (옵션) YAML 기준 디렉토리가 있다면 함께 붙여서 절대경로로 만들 수 있음

    Returns: callable
    """
    # 1) parse
    if ":" not in function_path:
        raise ValueError(f"Invalid function_path '{function_path}'. Must be 'xxx:func'")
    mod_part, func_name = function_path.rsplit(":", 1)

    mod_part = get_abspath(mod_part, base_dir).replace(".", "/")
    mod_part_fs = mod_part + ".py"
    mod_file = os.path.abspath(mod_part_fs)

    if not os.path.isfile(mod_file):
        raise FileNotFoundError(f"Cannot find file for module '{mod_part}': {mod_file}")

    # 3) use importlib.util to load .py file
    spec = importlib.util.spec_from_file_location("temp_loaded_module", mod_file)
    if not spec or not spec.loader:
        raise ImportError(f"Failed to create spec for '{mod_file}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4) get the target function
    if not hasattr(module, func_name):
        raise AttributeError(f"Module '{mod_file}' has no attribute '{func_name}'")
    func = getattr(module, func_name)

    if not callable(func):
        raise TypeError(f"Attribute '{func_name}' is not callable in '{mod_file}'")

    return func


class FunctionFromFileNode(BaseNode):
    """
    - function_path: "mypackage.module:some_func" 또는 "../test_funcs.my_module:func_name"처럼
      파일 경로를 '.'로 구분해 쓰는 방식도 허용.
    - input_keys: 함수에 넘길 state 키 목록
    - output_key: str or List[str] (반드시 존재)
    - params: 추가 인자 (state에 없는 값도 넣을 수 있음)
    - build()에서 함수 로딩 후, 호출 시 반환값(딕셔너리) 검증
    """

    def __init__(
        self,
        name: str = None,
        function_path: str = None,
        input_keys: List[str] = None,
        output_key: Union[str, List[str]] = None,
        base_dir: str = None,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name)
        self.function_path = function_path
        self.input_keys = input_keys
        self.params = params or {}
        self.base_dir = base_dir

        self.output_key = output_key
        self._func = None  # build() 시점에 임포트

    @staticmethod
    def from_yaml(config: dict, base_dir: str = None) -> "FunctionFromFileNode":
        """
        YAML 예시:
        - name: my_node
          type: function
          function_path: ../test_funcs.test_funcs_for_validation:partial_values_func
          input_keys: [a, b]
          output_key: "result"
          params:
            prefix: "[Test]"
        """
        if "output_key" not in config:
            raise ValueError(
                f"Missing 'output_key' in FunctionNode config: {config.get('name')}"
            )

        assert base_dir is not None

        return FunctionFromFileNode(
            name=config["name"],
            input_keys=config.get("input_keys", []),
            output_key=config["output_key"],
            function_path=config["config"]["function_path"],
            params=config["config"].get("params", {}),
            base_dir=base_dir,
        )

    def build(self):
        # (1) 함수 임포트 (상대 경로 -> 파일 로딩)
        self._func = load_function_from_path(self.function_path, base_dir=self.base_dir)

        # (2) 노드가 실행될 때 호출되는 함수(Closure)를 반환
        def node_fn(state: Dict) -> Dict:
            # 입력값 준비
            inputs = self.get_inputs(state)
            inputs.update(self.params)

            # 함수 호출 (dict 반환 필수)
            result = self._func(**inputs)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Function '{self.function_path}' must return a dict, "
                    f"but got: {type(result).__name__}"
                )

            # 결과 검증
            self._validate_result(result)
            return result

        return node_fn

    def _validate_result(self, result: dict):
        """
        함수 반환 dict의 키가 'output_key'(단일/리스트)와 정확히 일치하는지 검증.
        """
        if isinstance(self.output_key, str):
            expected_keys = {self.output_key}
        elif isinstance(self.output_key, list):
            expected_keys = set(self.output_key)
        else:
            raise ValueError(
                f"Invalid output_key type {type(self.output_key).__name__}, "
                "must be str or list[str]."
            )

        actual_keys = set(result.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Function '{self.function_path}' returned keys {actual_keys}, "
                f"but output_key expects exactly {expected_keys}."
            )


if __name__ == "__main__":
    from test_functions import test_function
    from agentblock.schema.tools import validate_yaml

    path_yaml = "template_function.yaml"
    validate_yaml(path_yaml)

    node_func = FunctionFromFileNode().from_yaml_file_single_node(path_yaml).build()
    result = node_func({"a": 1, "b": 2})

    result_expected = test_function(1, 2, x=1, y=2)
    assert (
        result["result"] == result_expected
    ), f"Expected: {result_expected}, Actual: {result}"

    from agentblock.graph_builder import GraphBuilder

    graph = GraphBuilder(path_yaml).build_graph()

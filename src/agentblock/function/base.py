# agentblock/function/function_node.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from agentblock.base import BaseNode


class FunctionNode(BaseNode, ABC):
    """
    최상위(추상) 클래스.
    - 'Python 함수를 호출하여 dict 반환'하는 노드의 공통 로직/인터페이스를 정의.
    """

    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: Union[str, List[str]],
    ):
        super().__init__(name)
        self.input_keys = input_keys
        self.output_key = output_key
        self._func = None

    @abstractmethod
    def parse_config(self, config: dict, base_dir: str = None):
        """
        각 서브클래스가 config를 해석하여
        - 함수 모듈/이름
        - 추가 인자
        등을 파싱.
        """

    @abstractmethod
    def import_target_function(self):
        """
        서브클래스가 원하는 방식으로 (예: file_path vs library)
        실제 Python 함수를 import하거나 준비.
        """

    @abstractmethod
    def call_target_function(self, inputs: Dict[str, Any]) -> Any:
        """
        함수 호출 & raw_result 반환 (dict가 아닐 수 있음).
        """

    def build(self):
        """
        build()에서 서브클래스 로직을 순서대로 호출:
        1) parse_config(...)
        2) import_target_function()
        3) define node_fn(...)
        """
        self._prepare()

        def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
            # 1) 입력을 모으고 함수 호출
            inputs = self.get_inputs(state)
            raw_result = self.call_target_function(inputs)

            # 2) 결과를 dict로 변환
            return self._wrap_result(raw_result)

        return node_fn

    def _prepare(self):
        """
        공통: parse_config + import_target_function
        """
        self.parse_config(config={}, base_dir=None)  # 서브클래스가 오버라이드해서 실제 config를 잡아야 함
        self.import_target_function()

    def _wrap_result(self, raw_result: Any) -> Dict[str, Any]:
        """
        output_key가 단일/리스트인지에 따라 dict로 감싸거나 검증
        """
        if isinstance(self.output_key, str):
            return {self.output_key: raw_result}
        elif isinstance(self.output_key, list):
            # 여러 키인 경우 raw_result가 동일 길이 tuple/list
            if not isinstance(raw_result, (list, tuple)) or len(raw_result) != len(
                self.output_key
            ):
                raise ValueError(
                    f"Expected multiple outputs {self.output_key}, "
                    f"but function returned {raw_result} with len mismatch."
                )
            return dict(zip(self.output_key, raw_result))
        else:
            raise ValueError(f"Invalid output_key: {self.output_key}")

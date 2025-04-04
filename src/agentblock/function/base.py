# agentblock/function/function_node.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, TypeVar, Generic
from dataclasses import dataclass
from agentblock.base import BaseNode

T = TypeVar('T')

@dataclass
class FunctionResult(Generic[T]):
    """함수 실행 결과를 감싸는 데이터 클래스"""
    value: T
    metadata: Optional[Dict[str, Any]] = None


class FunctionNode(BaseNode, ABC, Generic[T]):
    """
    최상위(추상) 클래스.
    - 'Python 함수를 호출하여 dict 반환'하는 노드의 공통 로직/인터페이스를 정의.
    """

    def __init__(
        self,
        name: str,
        input_keys: List[str],
        output_key: Union[str, List[str]],
        validate_inputs: bool = True,
        validate_outputs: bool = True,
    ):
        super().__init__(name)
        self.input_keys = input_keys
        self.output_key = output_key
        self.validate_inputs = validate_inputs
        self.validate_outputs = validate_outputs
        self._func = None

    @abstractmethod
    def parse_config(self, config: dict, base_dir: Optional[str] = None) -> None:
        """
        각 서브클래스가 config를 해석하여
        - 함수 모듈/이름
        - 추가 인자
        등을 파싱.
        """
        pass

    @abstractmethod
    def import_target_function(self) -> None:
        """
        서브클래스가 원하는 방식으로 (예: file_path vs library)
        실제 Python 함수를 import하거나 준비.
        """
        pass

    @abstractmethod
    def call_target_function(self, inputs: Dict[str, Any]) -> FunctionResult[T]:
        """
        함수 호출 & FunctionResult 반환.
        """
        pass

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """입력값 검증"""
        if not self.validate_inputs:
            return
            
        # 매핑된 키를 파싱하여 검증
        expected_keys = set()
        for k in self.input_keys:
            _, dest_key = self.parse_input_keys(k)
            expected_keys.add(dest_key)

        missing_keys = expected_keys - set(inputs.keys())
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")

    def _validate_output(self, result: FunctionResult[T]) -> None:
        """출력값 검증"""
        if not self.validate_outputs:
            return
            
        if not isinstance(result, FunctionResult):
            raise TypeError(f"Expected FunctionResult, got {type(result)}")

    def build(self):
        """
        build()에서 서브클래스 로직을 순서대로 호출:
        1) parse_config(...)
        2) import_target_function()
        3) define node_fn(...)
        """
        self._prepare()

        def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # 1) 입력을 모으고 함수 호출
                inputs = self.get_inputs(state)
                self._validate_inputs(inputs)
                
                # 2) 함수 실행 및 결과 검증
                result = self.call_target_function(inputs)
                self._validate_output(result)

                # 3) 결과를 dict로 변환
                return self._wrap_result(result.value)
            except Exception as e:
                # 원래 예외 타입을 유지하면서 메시지만 노드 정보를 추가
                e.args = (f"Error in node {self.name}: {str(e)}",) + e.args[1:]
                raise

        return node_fn

    def _prepare(self) -> None:
        """
        공통: parse_config + import_target_function
        """
        self.parse_config(config={}, base_dir=None)
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

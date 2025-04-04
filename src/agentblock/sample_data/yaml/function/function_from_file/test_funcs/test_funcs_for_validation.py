from typing import Dict, Any

from agentblock.function.base import FunctionResult

"""
test_funcs_for_validation.py

이 모듈에는 FunctionNode의 결과 검증을 테스트하기 위한
다양한 함수들이 포함되어 있습니다.

예를 들어:
- FunctionNode._validate_result 로직 등을 테스트할 수 있음.
- FunctionNode._wrap_result 로직 등을 테스트할 수 있음.
"""


def return_dict() -> Dict[str, Any]:
    """기본 dict 반환"""
    return {"key": "value"}


def return_list() -> list:
    """list 반환"""
    return [1, 2, 3]


def return_tuple() -> tuple:
    """tuple 반환"""
    return (1, 2)


def return_int() -> int:
    """int 반환"""
    return 42


def return_str() -> str:
    """str 반환"""
    return "hello"


def return_none() -> None:
    """None 반환"""
    return None


def return_function_result() -> FunctionResult[Dict[str, Any]]:
    """FunctionResult 반환"""
    return FunctionResult(
        value={"key": "value"},
        metadata={"source": "test"}
    )


def multiple_values_func(a: int, b: int) -> dict:
    """
    Returns 3 keys: sum, diff, product.
    { "sum": a+b, "diff": a-b, "product": a*b }
    """
    return a + b, a - b, a * b


def partial_values_func(a: int, b: int) -> dict:
    """
    Returns only 2 keys: sum, diff => missing 'product'.
    { "sum":..., "diff":... }
    """
    return a + b, a - b


def error_return_non_dict(data: str):
    """
    Returns a string instead of dict =>
    if FunctionNode expects dict, this is a violation.
    """
    return f"string_instead_of_dict:{data}"

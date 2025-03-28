"""
test_funcs_for_validation.py

이 모듈에는 FunctionNode의 결과 검증을 테스트하기 위한
여러 개의 함수를 정의합니다.

모두 dict를 반환(또는 dict가 아니게 해서 에러)하여
FunctionNode._validate_result 로직 등을 테스트할 수 있음.
"""


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

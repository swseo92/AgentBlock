def single_value_func(x: int, scale: int = 1):
    """
    예: 단일 숫자를 계산 → dict에 { 'value': result } 형태로 감싸서 반환
    """
    result = x * scale
    return result

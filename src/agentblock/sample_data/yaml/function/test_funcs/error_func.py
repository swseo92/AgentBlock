def error_func(data: str):
    """
    dict가 아닌 int/str 등을 반환해서 오류를 유발한다.
    """
    # 그냥 문자열 반환 (dict 아님) => FunctionNode 정책과 충돌
    return f"Non-dict result: {data}"

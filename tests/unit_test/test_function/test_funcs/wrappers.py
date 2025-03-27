def original_int_function(x, y):
    """
    원래 int만 반환하는 예시
    """
    return x + y


def wrapped_int_function(x, y):
    """
    original_int_function()을 감싸 dict로 반환
    """
    val = original_int_function(x, y)
    return {"wrapped_result": val}

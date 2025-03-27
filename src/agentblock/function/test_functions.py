def test_function(a, b, x=1, y=2):
    result = a * x + b * y
    return result


def test_function_wrapper(a, b, x=1, y=2):
    result = test_function(a, b, x=x, y=y)
    return {"result": result}  # yaml에 정의 된 output_key를 포함

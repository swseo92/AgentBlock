def sum_cal(a, b, x=0, y=0):
    return a + b + x + y


def multi_ops(a, b, x=0, y=0):
    s = a + b + x + y
    d = a - b
    mul = a * b + x * y

    div = a / b if b != 0 else None
    return (s, d, mul, div)


def short_ops(a, b):
    return (a + b, a * b)


def debug_sum(a, x=0):
    return a + x


def risky_div(a, b):
    return a / b  # b=0 => ZeroDivisionError


def none_func(a, b):
    # 단순히 None 반환
    return None

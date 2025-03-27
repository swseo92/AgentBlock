import pytest
from agentblock.graph_builder import GraphBuilder
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


base_path = f"{script_dir}/test_yaml/"


def test_single_value():
    """
    Test the single_value_func with scale=2 => result['value'] = x*2
    """
    yaml_path = f"{base_path}/test_single_value.yaml"
    graph = GraphBuilder(yaml_path).build_graph()

    result = graph.invoke({"x": 10})
    # single_value_func => {"value": 10*2=20}
    assert result["value"] == 20, result


def test_multiple_values():
    """
    multiple_values_func => {sum, diff, product}
    """
    # BUGFIX: changed "f{base_path}" → f"{base_path}"
    yaml_path = f"{base_path}/test_multiple_values.yaml"
    graph = GraphBuilder(yaml_path).build_graph()

    result = graph.invoke({"a": 5, "b": 2})
    # multiple_values_func => {"sum":7, "diff":3, "product":10}
    assert result["sum"] == 7, result
    assert result["diff"] == 3, result
    assert result["product"] == 10, result


def test_wrapped_func():
    """
    wrapped_return_dict => {"wrapped_result": x+y}
    """
    yaml_path = f"{base_path}/test_wrapped_func.yaml"
    graph = GraphBuilder(yaml_path).build_graph()

    result = graph.invoke({"x": 3, "y": 4})
    # => {"wrapped_result": 7}
    assert result["wrapped_result"] == 7


def test_error_func():
    """
    error_func => returns string => Expect ExecutionError
    """
    yaml_path = f"{base_path}/test_error_func.yaml"
    graph = GraphBuilder(yaml_path).build_graph()

    # 만약 FunctionNode 쪽에서 dict가 아닌 값이 반환되면
    # ValueError -> ExecutionError(...)로 래핑될 수 있음
    with pytest.raises(Exception) as exc_info:
        graph.invoke({"data": "Hello"})
        print(exc_info)

    # partial_state = exc_info.value.partial_state  # ExecutionError 사용 시
    # print("Partial state so far:", partial_state)
    # cause = exc_info.value.cause
    # assert "must return a dict" in str(cause)

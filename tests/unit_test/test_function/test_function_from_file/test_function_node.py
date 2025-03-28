from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


base_path = "yaml/function/function_from_file/test_yaml"
yaml_path_test_single_value = get_sample_data(f"{base_path}/test_single_value.yaml")
yaml_path_test_multiple_values = get_sample_data(
    f"{base_path}/test_multiple_values.yaml"
)


def test_single_value():
    """
    Test the single_value_func with scale=2 => result['value'] = x*2
    """
    graph = GraphBuilder(yaml_path_test_single_value).build_graph()

    result = graph.invoke({"x": 10})
    # single_value_func => {"value": 10*2=20}
    assert result["value"] == 20, result


def test_multiple_values():
    """
    multiple_values_func => {sum, diff, product}
    """
    # BUGFIX: changed "f{base_path}" → f"{base_path}"
    graph = GraphBuilder(yaml_path_test_multiple_values).build_graph()

    result = graph.invoke({"a": 5, "b": 2})
    # multiple_values_func => {"sum":7, "diff":3, "product":10}
    assert result["sum"] == 7, result
    assert result["diff"] == 3, result
    assert result["product"] == 10, result

import pytest
from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


base_path = "yaml/function/test_yaml"
yaml_path_test_single_key_extra = get_sample_data(f"{base_path}/single_key_extra.yaml")
yaml_path_test_multi_key_missing = get_sample_data(
    f"{base_path}/multi_key_missing.yaml"
)


def test_single_key_extra():
    """
    single_key_extra.yaml -> single_value_extra_func returns {'result':..., 'extra':999}
    but output_key='result' => mismatch => ValueError => ExecutionError
    """
    builder = GraphBuilder(yaml_path_test_single_key_extra)
    graph = builder.build_graph()

    with pytest.raises(Exception) as exc_info:
        graph.invoke({"x": 4})
        print(exc_info)
    # partial_state can be checked if needed


def test_multi_key_missing():
    """
    partial_keys => {sum,diff}, missing 'product' => mismatch => error
    """
    builder = GraphBuilder(yaml_path_test_multi_key_missing)
    graph = builder.build_graph()

    with pytest.raises(Exception):
        graph.invoke({"a": 5, "b": 2})

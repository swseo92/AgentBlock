import pytest
from agentblock.graph_builder import GraphBuilder
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


base_path = f"{script_dir}/test_yaml/"


def test_single_key_extra():
    """
    single_key_extra.yaml -> single_value_extra_func returns {'result':..., 'extra':999}
    but output_key='result' => mismatch => ValueError => ExecutionError
    """
    path = f"{base_path}/single_key_extra.yaml"
    builder = GraphBuilder(path)
    graph = builder.build_graph()

    with pytest.raises(Exception) as exc_info:
        graph.invoke({"x": 4})
        print(exc_info)
    # partial_state can be checked if needed


def test_multi_key_missing():
    """
    partial_keys => {sum,diff}, missing 'product' => mismatch => error
    """
    path = f"{base_path}/multi_key_missing.yaml"
    builder = GraphBuilder(path)
    graph = builder.build_graph()

    with pytest.raises(Exception):
        graph.invoke({"a": 5, "b": 2})

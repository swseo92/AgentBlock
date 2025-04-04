import pytest
from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


base_path = "yaml/function/function_from_file/test_yaml"
yaml_path_test_multi_key_missing = get_sample_data(
    f"{base_path}/multi_key_missing.yaml"
)


def test_multi_key_missing():
    """
    partial_keys => {sum,diff}, missing 'product' => mismatch => error
    """
    builder = GraphBuilder(yaml_path_test_multi_key_missing)
    graph = builder.build()

    with pytest.raises(Exception):
        graph.invoke({"a": 5, "b": 2})

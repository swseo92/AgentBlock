import os
import pytest
import yaml

from agentblock.graph_builder import GraphBuilder


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


BASE_PATH = f"{script_dir}/test_yaml/"  # 예: 테스트 YAML 폴더


def test_valid_config():
    """정상 케이스: validate_config 통과."""
    yaml_path = os.path.join(BASE_PATH, "test_valid_config.yaml")
    GraphBuilder(yaml_path)


@pytest.mark.parametrize(
    "filename",
    [
        "test_missing_nodes.yaml",
        "test_node_missing_name.yaml",
        "test_node_missing_type.yaml",
        "test_edges_not_list.yaml",
        "test_multiple_end.yaml",
        "test_unreachable_node.yaml",
        "test_node_cannot_reach_end.yaml",
        "test_missing_from_file_from_yaml.yaml",
    ],
)
def test_config_errors(filename):
    yaml_path = os.path.join(BASE_PATH, filename)
    with pytest.raises(ValueError) as excinfo:
        _ = GraphBuilder(yaml_path)
        print(excinfo)
    # 여기서 발생한 ValueError 메시지를 검사할 수도 있음
    # 예) assert "More than one edge leads to 'END'." in str(excinfo.value)

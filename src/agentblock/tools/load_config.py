import yaml
from pathlib import Path


def load_config(yaml_path: str) -> dict:
    """
    주어진 yaml 파일을 열어 dict 형태로 파싱하여 반환합니다.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_abspath(path: str, base_dir: str) -> str:
    base = Path(base_dir)
    target = base / Path(path)
    return str(target.resolve())


def get_parent_dir_abspath(file_path: str) -> str:
    return str(Path(file_path).resolve().parent)


def get_yaml_for_single_node_file(yaml_path):
    config_data = load_config(yaml_path)

    assert len(config_data["nodes"]) == 1
    return config_data["nodes"][0]


def get_yaml_for_single_reference_file(yaml_path):
    config_data = load_config(yaml_path)

    assert len(config_data["reference"]) == 1
    return config_data["reference"][0]

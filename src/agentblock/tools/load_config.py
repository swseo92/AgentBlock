import yaml


def load_config(yaml_path: str) -> dict:
    """
    주어진 yaml 파일을 열어 dict 형태로 파싱하여 반환합니다.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return config_dict

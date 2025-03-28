import os
from agentblock.schema.tools import validate_yaml


def get_sample_data(relative_path: str) -> str:
    """
    name에 해당하는 샘플 파일의 절대경로를 반환한다.
    get_sample_data 함수가 정의된 스크립트를 기준으로 상대경로를 매핑.
    """
    # 1) get_sample_data 함수가 들어 있는 파일(__file__)의 절대경로를 얻음
    this_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(this_file)

    # 3) 상대경로 → 절대경로
    absolute_path = os.path.join(base_dir, relative_path)
    absolute_path = os.path.abspath(absolute_path)

    # 4) 파일 존재 여부 검사
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Sample data file not found: '{absolute_path}'")

    # 5) 유효성 검사
    validate_yaml(absolute_path)
    return absolute_path

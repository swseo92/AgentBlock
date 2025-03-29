import os
import pytest

from agentblock.sample_data.tools import get_sample_data


def test_get_sample_data_file_found():
    """
    data_map 안에 있고, 실제 파일도 존재하는 키를 테스트합니다.
    (e.g. dummy_embedding).
    """
    # 1) 파일 존재 확인
    #    실제로 data_map["dummy_embedding"]가 가리키는 파일이
    #    프로젝트 내에 있어야 합니다.
    path = get_sample_data("yaml/embedding/reference/dummy_embedding.yaml")

    # 2) 반환값이 절대경로인지, 실제 파일이 존재하는지 체크
    assert os.path.isabs(path), "반환된 경로는 절대경로여야 합니다."
    assert os.path.exists(path), f"실제 파일이 존재해야 합니다: {path}"

    # 필요시 더 자세한 검사
    # 예: assert path.endswith("dummy_embedding.yaml"), "파일명이 일치해야 합니다."


def test_get_sample_data_file_not_found():
    """
    data_map 안에 있지만, 실제 파일이 존재하지 않는 경우 FileNotFoundError가 발생해야 합니다.

    예시:
     1) data_map에 "missing_file"이라는 키를 추가해, 실제 존재하지 않는 파일로 매핑
        "missing_file": "yaml/embedding/no_such_file.yaml"
     2) 그 뒤 여기서 "missing_file"로 호출
    """
    # 만약 이미 data_map에 없는 키면 ValueError이므로,
    # 'missing_file' 등으로 data_map에 등록해두고 실제 파일을 안 만들면 됨.

    with pytest.raises(FileNotFoundError):
        get_sample_data("missing_file")

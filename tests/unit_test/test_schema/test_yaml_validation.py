import pytest
from agentblock.schema.tools import validate_yaml
from agentblock.sample_data.tools import get_sample_data


base_path_valid = "yaml/schema/valid_case"
base_path_invalid = "yaml/schema/invalid_case"

list_filename_valid = ["good_schema.yaml", "references_only.yaml"]


def get_path_invalid(filename):
    return get_sample_data(f"{base_path_invalid}/{filename}")


#
# 1) 정상 케이스 (PASS) 목록
#
@pytest.mark.parametrize(
    "yaml_file",
    [get_sample_data(f"{base_path_valid}/{path}") for path in list_filename_valid],
)
def test_validate_yaml_success_cases(yaml_file):
    """
    정상 케이스: 에러 없이 통과해야 함
    """
    validate_yaml(yaml_file)


#
# 2) 실패 케이스 (FAIL) 목록
#
@pytest.mark.parametrize(
    "yaml_file, err_msg_regex",
    [
        ("bad_schema.yaml", "name.*없거나.*"),  # 노드에 name 필드 누락
        ("duplicate_names.yaml", "중복되었습니다"),  # refs & nodes 이름 겹침
        ("non_execution_in_nodes.yaml", "비실행 노드.*허용되지 않습니다"),
        ("missing_start_edge.yaml", "START에서 시작하는 edge가 없습니다"),
        ("missing_end_edge.yaml", "END로 가는 edge가 없습니다"),
        ("multiple_end_edges.yaml", "END로 향하는 edge가.*존재합니다"),
        ("invalid_field.yaml", "최상위에서 허용되지 않은 필드가 발견되었습니다"),
        ("not_list_edges.yaml", "edges' 필드는 list 형식이어야 합니다."),
        ("not_list_nodes.yaml", "nodes' 필드는 list 형식이어야 합니다."),
        ("not_list_references.yaml", "references' 필드는 list 형식이어야 합니다."),
    ],
)
def test_validate_yaml_failure_cases(yaml_file, err_msg_regex):
    """
    실패 케이스: 특정 에러 메시지와 match되어야 함
    """
    with pytest.raises(ValueError, match=err_msg_regex):
        get_path_invalid(yaml_file)

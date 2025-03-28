import pytest
from pathlib import Path
from agentblock.data_loader.base import GenericLoaderNode
from agentblock.sample_data.tools import get_sample_data
from agentblock.tools.load_config import load_config


@pytest.fixture
def sample_text_file(tmp_path: Path):
    """
    Pytest fixture: 임시 폴더에 텍스트 파일을 생성하고 경로를 리턴
    """
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("Hello from the file loader")
    return test_file


def test_generic_loader_node_without_graphbuilder(sample_text_file):
    """
    GraphBuilder 사용 없이, GenericLoaderNode를 직접 생성해 파일 로딩 기능을 테스트합니다.
    """
    # 1) 노드 설정(딕셔너리) 준비
    path_yaml = get_sample_data("yaml/data_loader/simple_loader.yaml")
    node_config = load_config(path_yaml)["nodes"][0]

    # 2) 노드 인스턴스 생성
    node = GenericLoaderNode.from_yaml(node_config)

    # 3) 테스트 실행할 때의 state 구성
    state = {"file_path": str(sample_text_file)}

    # 4) invoke() 호출
    new_state = node.invoke(state)
    docs = new_state["documents"]

    # 5) 검증
    assert len(docs) == 1
    assert docs[0].page_content == "Hello from the file loader"
    assert docs[0].metadata["source"] == str(sample_text_file)

    print("Test passed! Documents:", docs)

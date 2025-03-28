from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


path_yaml = get_sample_data("yaml/data_loader/simple_loader.yaml")


def test_data_loaders_with_yaml(tmp_path):
    # 1) 임시 파일 생성 (파일 로더용)
    test_file = tmp_path / "test_input.txt"
    test_file.write_text("Hello from file loader")

    # 2) YAML 로부터 그래프 빌드
    builder = GraphBuilder(path_yaml)
    graph = builder.build_graph()

    # 3) state 에 file_path 전달
    init_state = {"file_path": str(test_file)}
    final_state = graph.invoke(init_state)

    # 4) 결과 검증
    docs = final_state["documents"]

    assert len(docs) == 1
    assert "Hello from file loader" in docs[0].page_content
    print("Documents:", docs)

from langchain.schema import Document
from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


def test_graph_builder_integration(tmp_path):
    """
    통합 테스트:
    - faiss_node_test.yaml 내용을 임시 파일에 기록합니다.
    - GraphBuilder를 통해 그래프를 빌드하고,
      입력 state("documents": [문서])를 전달하여 DataSaverNode의 저장 결과를 확인합니다.
    """
    path_yaml = get_sample_data("yaml/vector_store/node/faiss_node_test.yaml")

    # GraphBuilder를 통해 그래프를 빌드합니다.
    builder = GraphBuilder(path_yaml)
    graph = builder.build_graph()

    # 입력 state 준비: "documents" 키에 단일 문서 입력
    state = {"documents": [Document(page_content="doc1")]}

    # 그래프 실행: 노드들을 순차적으로 실행하여 최종 state를 반환합니다.
    result_state = graph.invoke(state)

    # 결과 검증: DataSaverNode의 결과가 "result" 키에 저장되어 있어야 합니다.
    assert (
        "result" in result_state
    ), "Graph state should contain 'result' key from DataSaverNode."
    saver_result = result_state["result"]
    assert saver_result["status"] == "saved", "저장 상태가 'saved'여야 합니다."
    assert saver_result["num_docs"] == 1, "저장된 문서 수가 1이어야 합니다."

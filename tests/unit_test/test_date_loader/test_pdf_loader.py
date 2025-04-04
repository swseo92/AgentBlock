from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


path_yaml = get_sample_data("yaml/data_loader/pdf_loader.yaml")
path_pdf = get_sample_data("pdf/공매도 재개의 증시 영향.pdf")


def test_pdf_loader():
    # 3) 그래프 빌드
    builder = GraphBuilder(path_yaml)
    graph = builder.build()

    # 4) 실행
    init_state = {"file_path": path_pdf}
    result_state = graph.invoke(init_state)
    docs = result_state["pdf_documents"]

    # 5) 검증
    # PDF가 정말 내용이 있으면 docs에 여러 페이지가 들어있을 것
    # 여기서는 거의 빈 PDF이므로, docs 길이나 page_content가 비어있을 가능성도 있음
    print("PDF docs:", docs)
    # 예시 검증
    assert isinstance(docs, list)
    for d in docs:
        assert "source" in d.metadata
        assert d.metadata["source"] == path_pdf

    # 추가 검증 로직: 페이지 수나 page_content를 체크할 수도 있음

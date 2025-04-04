import os
import pytest
from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data


path_yaml = get_sample_data("graph/preprocessing_pipeline/simple_pipeline.yaml")
path_pdf = get_sample_data("pdf/공매도 재개의 증시 영향.pdf")


@pytest.mark.integration
def test_pdf_loader_with_character_text_split():
    """
    pdf_loader 노드로 PDF를 로드한 뒤,
    character_text_split으로 문서를 분할하는 통합 테스트.

    1) YAML 그래프를 빌드
    2) PDF 파일 경로(file_path)를 state에 넣어 graph.invoke() 호출
    3) split_docs 결과를 확인
    """

    # 1) GraphBuilder로 test_pdf_split.yaml 그래프 구성
    builder = GraphBuilder(path_yaml)
    graph = builder.build()

    # 2) PDF 파일 경로를 포함한 초기 state
    state = {"file_path": path_pdf}

    # 3) 그래프 실행
    output_state = graph.invoke(state)

    # 4) text_splitter 결과 확인
    split_docs = output_state.get("split_docs")
    assert split_docs is not None, "split_docs should exist in state"
    assert len(split_docs) > 0, "split_docs should not be empty"

    # 예시로 첫 번째 청크의 메타데이터나 내용 일부를 간단히 확인
    first_doc = split_docs[0]
    assert (
        "source" in first_doc.metadata
    ), "Metadata should contain 'source' indicating PDF path"
    assert (
        os.path.basename(path_pdf) in first_doc.metadata["source"]
    ), "Metadata should include the original PDF file path"

    print(f"Total splitted docs: {len(split_docs)}")
    print("Sample chunk:", first_doc.page_content[:100], "...")

    assert output_state["result"]["status"] == "saved"
    assert os.path.exists(output_state["result"]["path_save"])

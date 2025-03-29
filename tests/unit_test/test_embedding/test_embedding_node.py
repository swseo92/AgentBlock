import pytest
from langchain.schema import Document
from agentblock.embedding.embedding_node import EmbeddingNode
from agentblock.embedding.dummy_embedding import DummyEmbedding
from agentblock.tools.load_config import load_config
from agentblock.sample_data.tools import get_sample_data


def setup_embedding_node_from_yaml(method="embed_documents"):
    path_yaml = get_sample_data("yaml/embedding/node/dummy_embedding.yaml")
    yaml_data = load_config(path_yaml)

    # Dummy 임베딩 객체 생성
    embedding_reference = DummyEmbedding(dimension=5)

    reference_map = {"dummy_embedding_reference": embedding_reference}

    # EmbeddingNode 객체 생성
    node_config = yaml_data["nodes"][0]
    node_config["config"]["param"]["method"] = method
    node = EmbeddingNode.from_yaml(
        node_config, base_dir="", references_map=reference_map
    )
    return node, embedding_reference


def test_embedding_node_with_list_of_documents():
    node, embedding_reference = setup_embedding_node_from_yaml(method="embed_documents")
    node2, embedding_reference2 = setup_embedding_node_from_yaml(method="embed_query")
    node3, embedding_reference3 = setup_embedding_node_from_yaml(method="__init__")

    # List[Document]로 입력값 생성
    docs = [Document(page_content="Document 1"), Document(page_content="Document 2")]

    # node_fn 생성
    node_fn = node.build()
    node_fn2 = node2.build()

    # 상태 값 (state) 전달
    state = {"raw_docs": docs, "raw_docs2": docs, "raw_docs3": docs}
    result = node_fn(state)
    result2 = node_fn2(state)

    assert result2["embedded_docs"] == result["embedded_docs"]

    with pytest.raises(ValueError):
        node3.build()(state)

    list_embedding = result["embedded_docs"][1]
    # 결과 검증: docs에 임베딩 벡터가 추가되었는지 확인
    assert list_embedding[0] == [0.1, 0.1, 0.1, 0.1, 0.1]
    assert list_embedding[1] == [0.1, 0.1, 0.1, 0.1, 0.1]
    assert len(list_embedding) == 2


def test_embedding_node_with_single_document():
    node, embedding_reference = setup_embedding_node_from_yaml()

    # 단일 Document로 입력값 생성
    docs = [Document(page_content="Single Document")]

    # node_fn 생성
    node_fn = node.build()

    # 상태 값 (state) 전달
    state = {"raw_docs": docs}
    result = node_fn(state)

    # 결과 검증: docs에 임베딩 벡터가 추가되었는지 확인
    assert result["embedded_docs"][1][0] == [0.1, 0.1, 0.1, 0.1, 0.1]


def test_invalid_method():
    node, embedding_reference = setup_embedding_node_from_yaml()

    # 존재하지 않는 메서드 호출 테스트
    node_config = {
        "name": "invalid",
        "input_keys": ["raw_docs"],
        "output_key": "embedded_docs",
        "config": {
            "name": "embedding_node",
            "param": {"method": "non_existent_method"},
            "reference": {"embedding": "dummy_embedding_reference"},
        },
    }

    reference_map = {"dummy_embedding_reference": embedding_reference}

    # EmbeddingNode 객체 생성

    node = EmbeddingNode.from_yaml(
        node_config, base_dir="", references_map=reference_map
    )
    with pytest.raises(ValueError):
        node.build()

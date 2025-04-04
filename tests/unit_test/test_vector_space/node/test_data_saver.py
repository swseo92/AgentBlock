import pytest
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore
from src.agentblock.vector_store.data_saver_node import DataSaverNode
from agentblock.vector_store.vector_store_reference import VectorStoreReference
from agentblock.embedding.embedding_reference import EmbeddingReference

from agentblock.tools.load_config import load_config
from agentblock.sample_data.tools import get_sample_data
from unittest.mock import MagicMock


path_yaml = get_sample_data("yaml/vector_store/node/faiss_node_test.yaml")


input_data = {
    "documents": [Document(page_content="doc1"), Document(page_content="doc2")]
}


def create_references_map():
    """
    DummyEmbedding과 in-memory FAISS 벡터 스토어를 생성하여
    references_map에 등록합니다.
    """
    references_map = {}
    config = load_config(path_yaml)

    embedding_config = config["references"][0]
    vector_store_config = config["references"][1]

    embedding_model = EmbeddingReference.from_yaml(embedding_config, ".", {})
    references_map["dummy_emb"] = embedding_model.build()

    vs_ref = VectorStoreReference.from_yaml(
        vector_store_config, base_dir=".", references_map=references_map
    )
    references_map["my_faiss"] = vs_ref.build()

    return references_map


def create_data_saver_yaml_config():
    """
    DataSaverNode의 YAML 설정 예시를 dict로 반환합니다.
    """
    config = load_config(path_yaml)
    return config["nodes"][0]


def test_data_saver_node_no_reference():
    """
    DataSaverNode가 입력 문서를 벡터 스토어에 올바르게 저장하는지 테스트합니다.
    tmp_path: pytest에서 제공하는 임시 디렉토리 경로.
    """
    # 임시 FAISS 인덱스 파일 경로 생성
    config = create_data_saver_yaml_config()
    with pytest.raises(ValueError, match="Reference .* not found"):
        DataSaverNode.from_yaml(config, base_dir=".", references_map={})


def test_data_saver_node_indexing():
    """
    DataSaverNode가 입력 문서를 벡터 스토어에 올바르게 저장하는지 테스트합니다.
    tmp_path: pytest에서 제공하는 임시 디렉토리 경로.
    """
    # 임시 FAISS 인덱스 파일 경로 생성
    references_map = create_references_map()
    config = create_data_saver_yaml_config()
    data_saver_node = DataSaverNode.from_yaml(
        config, base_dir=".", references_map=references_map
    )

    # "documents" 키에 문자열 리스트를 입력으로 전달 (자동으로 Document로 변환됨)
    result = data_saver_node.call_target_function(input_data)

    # 저장 결과 반환값 확인 (FunctionResult.value에 접근)
    assert result.value["status"] == "saved"
    assert result.value["num_docs"] == 2

    # 추가로, 저장된 문서가 실제 벡터 스토어에 반영되었는지 similarity_search로 검증합니다.
    vs = references_map["my_faiss"]
    search_results = vs.similarity_search("Test", k=10)
    # DummyEmbedding은 항상 고정된 벡터를 반환하므로, 입력한 문서들이 검색 결과에 포함되어야 합니다.
    assert len(search_results) >= 2


def test_data_saver_node_invalid_input():
    """
    'documents' 키가 없는 입력에 대해 예외가 발생하는지 테스트합니다.
    """
    references_map = create_references_map()
    config = create_data_saver_yaml_config()
    data_saver_node = DataSaverNode.from_yaml(
        config, base_dir=".", references_map=references_map
    )

    with pytest.raises(ValueError):
        # 'documents' 키가 없으면 ValueError가 발생해야 함
        data_saver_node.call_target_function({})


@pytest.fixture
def setup_data_saver_node():
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_vector_store.save = MagicMock()
    mock_vector_store.path_save = "mock_path"
    node = DataSaverNode(
        name="test_node",
        reference=mock_vector_store,
        input_keys=["documents"],
        output_key="result",
    )
    return node, mock_vector_store


def test_data_saver_node_with_valid_documents(setup_data_saver_node):
    node, mock_vector_store = setup_data_saver_node
    documents = [Document(page_content="doc1"), Document(page_content="doc2")]
    inputs = {"documents": documents}

    result = node.call_target_function(inputs)

    assert result.value["status"] == "saved"
    assert result.value["num_docs"] == 2
    mock_vector_store.add_documents.assert_called_once_with(documents)


def test_data_saver_node_with_invalid_document_type(setup_data_saver_node):
    node, _ = setup_data_saver_node
    inputs = {"documents": [123, {"key": "value"}]}

    with pytest.raises(ValueError, match="The inputs must be an instance of Document"):
        node.call_target_function(inputs)


def test_data_saver_node_with_missing_documents_key(setup_data_saver_node):
    node, _ = setup_data_saver_node
    inputs = {}

    with pytest.raises(ValueError, match="No documents to save."):
        node.call_target_function(inputs)

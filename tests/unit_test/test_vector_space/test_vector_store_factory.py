import os
import pytest
import tempfile
import yaml

from agentblock.vector_store.vector_store_factory import VectorStoreFactory
from langchain_core.embeddings.embeddings import Embeddings
from agentblock.embedding.dummy_embedding import DummyEmbedding

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv


load_dotenv()


@pytest.fixture
def embedding_model() -> Embeddings:
    return DummyEmbedding()


def test_create_vector_store_faiss(embedding_model):
    """
    1) create_vector_store()가 'faiss' provider일 때
       정상적으로 FAISS 스토어를 생성하는지 테스트.
    """
    factory = VectorStoreFactory()
    vector_store = factory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )

    # 반환된 객체가 FAISS 인스턴스인지 확인
    assert isinstance(vector_store, FAISS), "Expected a FAISS vector store instance"

    # 검색 시 아무 문서도 없으므로 결과 0개
    results = vector_store.similarity_search("test", k=1)
    assert len(results) == 0, "Empty store should have 0 results"


def test_create_vector_store_no_embedding():
    """
    2) embedding_model이 None인 경우, ValueError가 발생해야 함.
    """
    factory = VectorStoreFactory()
    with pytest.raises(ValueError) as exc_info:
        factory.create_vector_store(provider="faiss", embedding_model=None)
    assert "embedding_model must be provided" in str(exc_info.value)


def test_create_vector_store_unsupported_provider(embedding_model):
    """
    3) 지원하지 않는 provider를 넣으면 예외 발생.
    """
    factory = VectorStoreFactory()
    with pytest.raises(ValueError) as exc_info:
        factory.create_vector_store(
            provider="pinecone", embedding_model=embedding_model
        )
    assert "Unsupported vector store provider" in str(exc_info.value)


def test_create_from_yaml_faiss(embedding_model):
    """
    4) create_from_yaml()를 통해 파라미터를 yaml 파일로부터 읽어오는지 테스트.
       - provider = "faiss"
       - config:
         path: None  (새 인덱스 생성)
       실제로 문서를 추가/검색이 되는지 간단히 확인.
    """
    factory = VectorStoreFactory()

    test_config = {
        "vector_store": {
            "provider": "faiss",
            "config": {
                # path=None이면 새 인덱스 생성
                "path": None
                # 다른 파라미터 (top_k 등)을 추가해도 됩니다
            },
        }
    }

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(test_config, tmp_file)
        tmp_yaml_path = tmp_file.name

    try:
        # YAML에서 설정 읽어옴
        vector_store = factory.create_from_yaml(
            yaml_path=tmp_yaml_path, embedding_model=embedding_model
        )

        # 여기도 FAISS 인스턴스인지 확인
        assert isinstance(vector_store, FAISS)

        # 문서 추가 -> 검색 테스트
        docs = ["Hello VectorStore", "LangChain test doc"]
        vector_store.add_texts(docs)

        results = vector_store.similarity_search("Hello", k=2)
        assert len(results) > 0, "Expected at least one result for 'Hello'"
        contents = [r.page_content for r in results]
        assert "Hello VectorStore" in contents, "Expected the doc 'Hello VectorStore'"

    finally:
        # 임시 파일 정리
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)


def test_create_from_yaml_no_embedding():
    """
    5) create_from_yaml 시 embedding_model=None 이면 ValueError.
    """
    factory = VectorStoreFactory()
    test_config = {"vector_store": {"provider": "faiss", "config": {"path": None}}}

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(test_config, tmp_file)
        tmp_yaml_path = tmp_file.name

    try:
        with pytest.raises(ValueError) as exc_info:
            factory.create_from_yaml(yaml_path=tmp_yaml_path, embedding_model=None)
        assert len(str(exc_info.value)) > 0
    finally:
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)


def test_create_from_yaml_unsupported_provider(embedding_model):
    """
    6) YAML에 미지원 provider가 있으면 예외 발생.
    """
    factory = VectorStoreFactory()
    test_config = {"vector_store": {"provider": "unknown", "config": {}}}

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(test_config, tmp_file)
        tmp_yaml_path = tmp_file.name

    try:
        with pytest.raises(ValueError) as exc_info:
            factory.create_from_yaml(
                yaml_path=tmp_yaml_path, embedding_model=embedding_model
            )
        assert "Unsupported vector store provider" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_yaml_path):
            os.remove(tmp_yaml_path)

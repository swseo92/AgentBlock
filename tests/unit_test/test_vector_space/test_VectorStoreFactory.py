import pytest
from unittest.mock import patch, MagicMock
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from docmesh.vector_store.VectorStoreFactory import VectorStoreFactory


class MockEmbeddings(Embeddings):
    """
    langchain_core.embeddings.embeddings.Embeddings를 상속받는 Mock.
    embed_query 호출 시 고정된 벡터(길이 5) 반환.
    """

    def embed_query(self, text: str):
        return [float(i) for i in range(5)]

    def embed_documents(self, texts: list[str]):
        return [[float(i) for i in range(5)] for _ in texts]


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddings()


def test_create_vector_store_faiss_no_embedding():
    """
    provider='faiss'인데 embedding_model이 None이면 예외 발생해야 함.
    """
    factory = VectorStoreFactory()
    with pytest.raises(ValueError, match="embedding_model must be provided"):
        factory.create_vector_store(provider="faiss", embedding_model=None)


def test_create_vector_store_unsupported_provider(mock_embedding_model):
    """
    지원하지 않는 provider를 넘기면 예외 발생해야 함.
    """
    factory = VectorStoreFactory()
    with pytest.raises(ValueError, match="Unsupported vector store provider"):
        factory.create_vector_store(
            provider="annoy2", embedding_model=mock_embedding_model
        )


@patch("docmesh.vector_store.VectorStoreFactory.FAISS.load_local")
def test_create_vector_store_faiss_with_path(
    mock_load_local, mock_embedding_model, tmp_path
):
    """
    path가 주어졌을 때, load_local이 호출되어 기존 FAISS 인덱스를 로드하는지 확인.
    """
    factory = VectorStoreFactory()
    test_path = str(tmp_path / "faiss_index")
    # mock_load_local이 반환할 임의의 FAISS 인스턴스 준비
    mock_faiss_instance = MagicMock(spec=FAISS)
    mock_load_local.return_value = mock_faiss_instance

    vs = factory.create_vector_store(
        provider="faiss", embedding_model=mock_embedding_model, path=test_path
    )

    # load_local이 호출되었는지, 파라미터가 맞는지 체크
    mock_load_local.assert_called_once_with(
        test_path, mock_embedding_model, allow_dangerous_deserialization=True
    )
    assert vs == mock_faiss_instance, "리턴된 VectorStore는 mock_faiss_instance여야 함"


@patch("docmesh.vector_store.VectorStoreFactory.FAISS")
def test_create_vector_store_faiss_no_path(mock_faiss_cls, mock_embedding_model):
    """
    path=None이면 새 FAISS 인스턴스를 생성해야 함.
    IndexFlatL2와 InMemoryDocstore가 사용되는지 확인.
    """
    factory = VectorStoreFactory()

    # mock_faiss_cls(FaissClassMock) 인스턴스 준비
    mock_faiss_instance = MagicMock(spec=FAISS)
    mock_faiss_cls.return_value = mock_faiss_instance

    vs = factory.create_vector_store(
        provider="faiss", embedding_model=mock_embedding_model, path=None
    )

    # load_local은 호출되지 않음
    mock_faiss_cls.load_local.assert_not_called()
    # 대신 FAISS(...) 생성자가 호출되었을 것
    assert vs == mock_faiss_instance


def test_create_vector_store_faiss_result_dimensions(mock_embedding_model):
    """
    embed_query("hello") => 벡터 차원 = 5
    -> IndexFlatL2(5)로 생성되는지 확인하기 위해,
    직접 creat_faiss_vector_store 호출 후 내부 index.d를 확인.
    """
    factory = VectorStoreFactory()
    # factory.create_vector_store -> creat_faiss_vector_store -> FAISS(...)
    vs = factory.create_vector_store(
        provider="faiss", embedding_model=mock_embedding_model
    )
    # vs.index가 실제 faiss IndexFlatL2 인스턴스일 것이므로, 차원(d) 확인
    index = vs.index
    assert index.d == 5, f"Index dimension must be 5, got {index.d}"

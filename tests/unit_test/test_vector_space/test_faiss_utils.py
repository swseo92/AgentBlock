import os
import pytest
import tempfile
from langchain_core.embeddings.embeddings import Embeddings
from agentblock.embedding.dummy_embedding import DummyEmbedding

from agentblock.vector_store.faiss_utils import create_faiss_vector_store
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def embedding_model() -> Embeddings:
    return DummyEmbedding()


def test_faiss_store_creation(embedding_model):
    """
    1) 벡터 스토어가 정상 생성되는지 확인
       - path=None으로 새 인덱스를 만들고,
         빈 상태이므로 검색 결과가 없는지 체크
    """
    vector_store = create_faiss_vector_store(
        embedding_model=embedding_model, path=None  # 새로 생성
    )
    # 반환 타입이 FAISS 인스턴스인지 확인
    assert isinstance(vector_store, FAISS), "Should return a FAISS instance"

    # 아직 어떤 문서도 넣지 않았으므로, 검색 결과 = 0개
    results = vector_store.similarity_search("test", k=1)
    assert len(results) == 0, "Empty store should return no results."


def test_faiss_add_texts(embedding_model):
    """
    2) 문서 add가 정상적으로 되는지 확인
       - add_texts로 문서 추가 후, 검색했을 때 해당 문서가 나오는지
    """
    vector_store = create_faiss_vector_store(embedding_model=embedding_model, path=None)

    docs = ["Hello world", "Foo bar", "FAISS test doc"]
    vector_store.add_texts(docs)

    # 검색: "Hello"가 들어간 문서가 잘 나오는지
    results = vector_store.similarity_search("Hello", k=2)
    assert len(results) > 0, "Should retrieve at least 1 doc for 'Hello'"

    # 결과 중 'Hello world'가 있는지 확인
    retrieved_texts = [doc.page_content for doc in results]
    assert "Hello world" in retrieved_texts, "Expected 'Hello world' in search results."


def test_faiss_save_local(embedding_model):
    """
    3) 저장이 잘 작동하는지 확인
       - save_local로 인덱스를 파일로 저장
       - 해당 파일이 실제로 생성되었는지
    """
    vector_store = create_faiss_vector_store(embedding_model=embedding_model, path=None)
    vector_store.add_texts(["Doc A", "Doc B"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "faiss_index.bin")
        vector_store.save_local(save_path)

        # 파일이 생성되었는지 체크
        assert os.path.exists(save_path), f"FAISS index file not found at {save_path}"


def test_faiss_save_and_load(embedding_model):
    """
    4) 저장 후 로드했을 때 결과물이 같은지 확인
       - 문서를 추가하고 save_local
       - load_local로 새로 로드한 뒤, 동일 쿼리에 대해 같은 결과를 주는지 테스트
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "faiss_index.bin")

        vector_store_1 = create_faiss_vector_store(
            embedding_model=embedding_model, path=save_path
        )
        docs = ["Apple banana", "Orange fruit", "Computer device"]
        vector_store_1.add_texts(docs)
        vector_store_1.save()

        # 새로 로드
        vector_store_2 = create_faiss_vector_store(
            embedding_model=embedding_model, path=save_path
        )

        # 동일한 쿼리로 검색
        query = "banana"
        results_1 = vector_store_1.similarity_search(query, k=3)
        results_2 = vector_store_2.similarity_search(query, k=3)

        # 결과 개수 비교
        assert len(results_1) == len(
            results_2
        ), "Loaded store should return the same number of results"

        # 문서 내용 비교 (집합으로 변환해 순서 무관 비교)
        set1 = set(doc.page_content for doc in results_1)
        set2 = set(doc.page_content for doc in results_2)
        assert (
            set1 == set2
        ), "Loaded store should return the same documents as original store"

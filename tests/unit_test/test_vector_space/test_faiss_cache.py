import os
import pytest
import tempfile
import shutil

from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from agentblock.vector_store.vector_store_factory import VectorStoreFactory
from agentblock.sample_data.tools import get_sample_data


class SimpleEmbedding(Embeddings):
    """고정된 3차원 벡터를 반환하는 간단한 Embedding (API 없이 사용 가능)."""

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def __str__(self):
        return "SimpleEmbedding-3dim"


class AnotherEmbedding(Embeddings):
    """같은 차원이지만 __str__가 달라 -> 다른 임베딩 모델로 간주."""

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def __str__(self):
        return "AnotherEmbedding"


@pytest.fixture
def factory():
    """캐싱 로직이 구현된 VectorStoreFactory 인스턴스."""
    return VectorStoreFactory()


def test_caching_same_path_io(factory: VectorStoreFactory):
    """
    1) 실제 파일을 만들어둔 뒤, 동일 path + 동일 임베딩 -> 캐시 적용으로 동일 객체.
       - 첫번째 호출: 인덱스 로드 (또는 생성)
       - 두번째 호출: 동일 path -> 캐시된 객체 재사용
    """
    embedding = SimpleEmbedding()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_sample = get_sample_data("yaml/vector_store/faiss.yaml")
        path_test = os.path.join(tmp_dir, "faiss.yaml")
        shutil.copy(path_sample, path_test)

        # 우선 새 인덱스 생성 (path=None) -> 문서 추가 -> save_local(index_path)
        store_init = factory.from_yaml_file_single_node(
            path_test, embedding_model=embedding
        )

        store_init.add_texts(["Test doc"])
        store_init.save()
        assert os.path.exists(store_init._path_save), "FAISS 인덱스 파일이 생성되어야 함"

        # 이제 vector_store.yaml 생성 (provider=faiss, config.path=index_path)
        # 첫 호출 - 실제로 index_path 로딩
        store_a = factory.from_yaml_file_single_node(
            path_test, embedding_model=embedding
        )
        assert isinstance(store_a, FAISS)
        # 검색 확인
        res_a = store_a.similarity_search("Test", k=1)
        assert len(res_a) == 1
        assert res_a[0].page_content == "Test doc"

        # 두 번째 호출 - 동일 path + 동일 embedding -> 캐시 hit
        store_b = factory.from_yaml_file_single_node(
            path_test, embedding_model=embedding
        )

        # store_a와 store_b는 같은 객체
        assert store_a is store_b, "Same path + same embedding => same in-memory object"


def test_caching_different_path_io(factory: VectorStoreFactory):
    """
    2) 파일을 두 개 만들어서, 서로 다른 path -> 서로 다른 인스턴스
    """
    emb = SimpleEmbedding()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_sample = get_sample_data("yaml/vector_store/faiss.yaml")
        path_test = os.path.join(tmp_dir, "faiss.yaml")

        os.mkdir(os.path.join(tmp_dir, "subdir"))
        path_test2 = os.path.join(tmp_dir, "subdir/faiss.yaml")

        shutil.copy(path_sample, path_test)
        shutil.copy(path_sample, path_test2)

        # 첫번째 인덱스 파일
        st1 = factory.from_yaml_file_single_node(path_test, embedding_model=emb)
        st1.add_texts(["Doc1"])
        st1.save()

        # 두번째 인덱스 파일
        st2 = factory.from_yaml_file_single_node(path_test2, embedding_model=emb)
        st2.add_texts(["Doc2"])
        st2.save()

        # 각각 로드
        store_a = factory.from_yaml_file_single_node(path_test, embedding_model=emb)
        store_b = factory.from_yaml_file_single_node(path_test2, embedding_model=emb)

        # 캐시 키 다름 -> 다른 객체
        assert store_a is not store_b


def test_caching_different_embedding_io(factory: VectorStoreFactory):
    """
    3) 동일한 path + 다른 embedding 모델 -> 캐시 키 다름 -> 새 인스턴스
    """
    embA = SimpleEmbedding()
    embB = AnotherEmbedding()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_sample = get_sample_data("yaml/vector_store/faiss.yaml")
        path_test = os.path.join(tmp_dir, "faiss.yaml")
        shutil.copy(path_sample, path_test)

        # 실제 파일 생성
        store_init = factory.from_yaml_file_single_node(path_test, embedding_model=embA)
        store_init.add_texts(["Shared doc"])
        store_init.save()

        # 첫 호출 (embA)
        store_a = factory.from_yaml_file_single_node(path_test, embedding_model=embA)
        # 두 번째 (embB)
        store_b = factory.from_yaml_file_single_node(path_test, embedding_model=embB)

        # 서로 다른 객체
        assert (
            store_a is not store_b
        ), "Same path but different embedding => not the same instance"

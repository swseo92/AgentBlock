# tests/test_vector_store_factory_cache_io.py

import os
import pytest
import tempfile
import yaml

from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from agentblock.vector_store.vector_store_factory import VectorStoreFactory
from agentblock.vector_store.faiss_utils import create_faiss_vector_store


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


def _make_vectorstore_yaml(path: str) -> str:
    """
    provider=faiss, config.path=path 형태의 YAML 파일을 임시 생성
    """
    config = {"vector_store": {"provider": "faiss", "config": {"path": path}}}
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
    with tmp:
        yaml.dump(config, tmp)
        return tmp.name


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
        # 실제 FAISS 인덱스 파일을 만들어두기
        index_path = os.path.join(tmp_dir, "real_faiss_index.bin")

        # 우선 새 인덱스 생성 (path=None) -> 문서 추가 -> save_local(index_path)
        store_init = create_faiss_vector_store(embedding_model=embedding, path=None)
        store_init.add_texts(["Test doc"])
        store_init.save_local(index_path)
        assert os.path.exists(index_path), "FAISS 인덱스 파일이 생성되어야 함"

        # 이제 vector_store.yaml 생성 (provider=faiss, config.path=index_path)
        yaml_path = _make_vectorstore_yaml(index_path)

        # 첫 호출 - 실제로 index_path 로딩
        store_a = factory.create_from_yaml(yaml_path, embedding_model=embedding)
        assert isinstance(store_a, FAISS)
        # 검색 확인
        res_a = store_a.similarity_search("Test", k=1)
        assert len(res_a) == 1
        assert res_a[0].page_content == "Test doc"

        # 두 번째 호출 - 동일 path + 동일 embedding -> 캐시 hit
        store_b = factory.create_from_yaml(yaml_path, embedding_model=embedding)

        # store_a와 store_b는 같은 객체
        assert store_a is store_b, "Same path + same embedding => same in-memory object"


def test_caching_different_path_io(factory: VectorStoreFactory):
    """
    2) 파일을 두 개 만들어서, 서로 다른 path -> 서로 다른 인스턴스
    """
    emb = SimpleEmbedding()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 첫번째 인덱스 파일
        path1 = os.path.join(tmp_dir, "faiss1.bin")
        st1 = create_faiss_vector_store(emb, path=None)
        st1.add_texts(["Doc1"])
        st1.save_local(path1)

        # 두번째 인덱스 파일
        path2 = os.path.join(tmp_dir, "faiss2.bin")
        st2 = create_faiss_vector_store(emb, path=None)
        st2.add_texts(["Doc2"])
        st2.save_local(path2)

        # YAML1, YAML2
        yaml1 = _make_vectorstore_yaml(path1)
        yaml2 = _make_vectorstore_yaml(path2)

        # 각각 로드
        store_a = factory.create_from_yaml(yaml1, embedding_model=emb)
        store_b = factory.create_from_yaml(yaml2, embedding_model=emb)

        # 캐시 키 다름 -> 다른 객체
        assert store_a is not store_b


def test_caching_different_embedding_io(factory: VectorStoreFactory):
    """
    3) 동일한 path + 다른 embedding 모델 -> 캐시 키 다름 -> 새 인스턴스
    """
    embA = SimpleEmbedding()
    embB = AnotherEmbedding()

    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = os.path.join(tmp_dir, "faiss_shared.bin")

        # 실제 파일 생성
        store_init = create_faiss_vector_store(embA, path=None)
        store_init.add_texts(["Shared doc"])
        store_init.save_local(index_path)

        # YAML
        yaml_path = _make_vectorstore_yaml(index_path)

        # 첫 호출 (embA)
        store_a = factory.create_from_yaml(yaml_path, embedding_model=embA)
        # 두 번째 (embB)
        store_b = factory.create_from_yaml(yaml_path, embedding_model=embB)

        # 서로 다른 객체
        assert (
            store_a is not store_b
        ), "Same path but different embedding => not the same instance"

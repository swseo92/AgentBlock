import pytest

from agentblock.embedding.embedding_reference import EmbeddingReference
from agentblock.vector_store.vector_store_reference import VectorStoreReference
from agentblock.sample_data.tools import get_sample_data
from agentblock.tools.load_config import load_config


@pytest.fixture
def faiss_yaml_path():
    path = get_sample_data("yaml/vector_store/reference/faiss_test.yaml")
    return path


def test_vector_store_reference_faiss(faiss_yaml_path, tmp_path):
    """
    faiss_test.yaml 로딩 -> dummy 임베딩 build -> my_faiss 벡터 스토어 build ->
    문서 add -> 검색 테스트
    """
    # 1) YAML 로딩
    data = load_config(faiss_yaml_path)
    ref_list = data.get("references", [])
    if not ref_list:
        pytest.fail("No 'reference' section found in faiss_test.yaml")

    # 이 딕셔너리가 GraphBuilder의 reference_map과 유사한 역할
    references_map = {}

    # 2) 먼저 Embedding 타입 레퍼런스(embedding)를 빌드해 langchain Embeddings 객체를 저장
    for ref_def in ref_list:
        if ref_def["type"] == "embedding":
            # EmbeddingReference.from_yaml -> build
            emb_ref = EmbeddingReference.from_yaml(ref_def, "..", {})
            embedding_obj = emb_ref.build()
            # reference_map에는 이미 build된 Embeddings 객체를 저장
            references_map[ref_def["name"]] = embedding_obj

    # 3) 이후 VectorStore 타입 레퍼런스(vector_store)를 빌드
    #    my_faiss가 embedding: "dummy_emb"를 참조한다고 가정
    for ref_def in ref_list:
        if ref_def["type"] == "vector_store":
            # VectorStoreReference.from_yaml( ... references_map=... )
            vs_ref = VectorStoreReference.from_yaml(ref_def, "..", references_map)
            # build() -> LangChain VectorStore
            vs_obj = vs_ref.build()
            assert vs_obj is not None, "Failed to build FAISS VectorStore"

            # 4) 문서 추가 & 검색 테스트
            vs_ref.add_documents(["Hello world", "Foo bar", "LangChain test"])
            results = vs_ref.search("hello", k=2)
            assert len(results) == 2
            # 검색 결과에 'Hello world'가 포함될 것으로 기대
            # FAISS 유사도 스코어 등은 dummy embedding이라도 결과 순서는 일정
            doc_texts = [r.page_content for r in results]
            # 간단히 'Hello world'가 들어있는지 확인
            assert any("Hello world" in t for t in doc_texts)

            # 만약 disk 저장(예: test.faiss) 확인 필요하면 tmp_path를 활용
            # vs_obj.save() -> vs_obj.save_local(...)
            # e.g. vs_obj.save_local(str(tmp_path / "test_index.faiss"))

            print(f"Search results: {doc_texts}")

from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from agentblock.embedding.dummy_embedding import DummyEmbedding
import agentblock.tools.load_config as load_config

import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


path_dummy_embedding = load_config.get_abspath(
    "../../yaml/embeddings/dummy_embedding.yaml", script_dir
)
assert os.path.exists(
    path_dummy_embedding
), f"Dummy embedding config file not found: {path_dummy_embedding}"


def test_factory_create_dummy_embedding():
    # 3) 팩토리 통해 임베딩 로드
    embedding_model = EmbeddingModelFactory.from_yaml_file(path_dummy_embedding)

    # 4) 타입 및 벡터 형태 검증
    assert isinstance(
        embedding_model, DummyEmbedding
    ), f"Expected DummyEmbedding, got {type(embedding_model)}"

    # dimension=5 인지 확인
    query_vec = embedding_model.embed_query("test query")
    assert len(query_vec) == 5, "DummyEmbedding dimension mismatch."
    for val in query_vec:
        assert val == 0.1, "DummyEmbedding vector values should be 0.1."

    # 5) 문서 벡터 테스트
    docs = ["doc1", "doc2"]
    doc_vecs = embedding_model.embed_documents(docs)
    assert len(doc_vecs) == 2, "Should return 2 vectors."
    for vec in doc_vecs:
        assert len(vec) == 5, "Document vector dimension mismatch."
        for v in vec:
            assert v == 0.1, "Document vector values should be 0.1."

    # Cleanup: temp_dir, etc. (pytest가 자동 정리하거나, 직접 삭제 가능)

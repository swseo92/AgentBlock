import pytest
from agentblock.embedding.dummy_embedding import DummyEmbedding


def test_dummy_embedding_query():
    """
    DummyEmbedding이 embed_query()에서
    dimension 길이의 벡터를 반환하는지 확인.
    모든 값이 0.1인지도 체크.
    """
    dim = 4
    dummy = DummyEmbedding(dimension=dim)
    vec = dummy.embed_query("hello world")

    assert len(vec) == dim, f"Expected vector length {dim}, got {len(vec)}."
    for val in vec:
        assert val == 0.1, f"Expected value 0.1, got {val}."


def test_dummy_embedding_documents():
    """
    embed_documents()가 여러 문서에 대해
    동일한 형태의 벡터 리스트를 반환하는지 검증.
    """
    dim = 3
    dummy = DummyEmbedding(dimension=dim)
    docs = ["doc1", "doc2", "doc3"]
    result = dummy.embed_documents(docs)

    # 문서 개수만큼 벡터가 생성되어야 함
    assert len(result) == len(docs), "Number of vectors must match number of docs."

    for vec in result:
        assert len(vec) == dim, f"Each vector should have length {dim}."
        for val in vec:
            assert val == 0.1, "DummyEmbedding should fill vectors with 0.1."


@pytest.mark.parametrize("dim", [1, 5, 10])
def test_dummy_embedding_various_dims(dim):
    """
    파라미터라이즈 기법으로, 다양한 dimension 값에서도
    embed_query, embed_documents가 올바르게 작동하는지 확인.
    """
    dummy = DummyEmbedding(dimension=dim)
    query_vec = dummy.embed_query("test query")

    assert len(query_vec) == dim, f"query vector length mismatch (expected {dim})"

    doc_vecs = dummy.embed_documents(["a", "b"])
    assert len(doc_vecs) == 2, "Should return 2 vectors for 2 docs."
    for vec in doc_vecs:
        assert len(vec) == dim, f"doc vector length mismatch (expected {dim})"

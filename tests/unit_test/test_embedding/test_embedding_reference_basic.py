import pytest
from langchain.embeddings import OpenAIEmbeddings
from src.agentblock.embedding.dummy_embedding import DummyEmbedding
from src.agentblock.embedding.embedding_reference import EmbeddingReference


@pytest.fixture
def sample_embedding_reference_openai():
    return EmbeddingReference(
        name="test_openai",
        provider="openai",
        config={"param": {"openai_api_key": "dummy_key"}},
    )


@pytest.fixture
def sample_embedding_reference_dummy():
    return DummyEmbedding()


def test_build_returns_openai_embedding(sample_embedding_reference_openai):
    embedding = sample_embedding_reference_openai.build()
    assert isinstance(embedding, OpenAIEmbeddings), "Expected OpenAIEmbeddings instance"


def test_build_raises_error_for_unsupported_provider():
    reference = EmbeddingReference(
        name="test_invalid", provider="invalid_provider", config={}
    )
    with pytest.raises(
        ValueError, match="Unsupported embedding provider: invalid_provider"
    ):
        reference.build()

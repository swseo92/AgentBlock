import pytest
from agentblock.embedding.EmbeddingModelFactory import EmbeddingModelFactory
from langchain_openai import OpenAIEmbeddings
from agentblock.tools.load_config import load_config

from dotenv import load_dotenv

path_config = "../test_config.yaml"


def test_create_openai_embedding_model():
    load_dotenv()
    config = load_config(path_config)

    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    assert isinstance(embeddings, OpenAIEmbeddings)


def test_embed_query_openai_embedding_model():
    load_dotenv()
    config = load_config(path_config)

    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    embeddings.embed_query("hello world")
    assert isinstance(embeddings, OpenAIEmbeddings)


def test_create_invalid_embedding_model():
    with pytest.raises(ValueError):
        _ = EmbeddingModelFactory.create_embedding_model(provider="invalid")

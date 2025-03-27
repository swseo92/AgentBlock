import pytest
from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from langchain_openai import OpenAIEmbeddings
from agentblock.tools.load_config import load_config

from dotenv import load_dotenv
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

path_config = "../test_config.yaml"
load_dotenv()


def test_create_openai_embedding_model():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    config = load_config(path_config)
    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    assert isinstance(embeddings, OpenAIEmbeddings)


def test_embed_query_openai_embedding_model():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    config = load_config(path_config)

    embeddings = EmbeddingModelFactory.create_embedding_model(
        **config["embedding_model"]
    )

    embeddings.embed_query("hello world")
    assert isinstance(embeddings, OpenAIEmbeddings)


def test_create_invalid_embedding_model():
    with pytest.raises(ValueError):
        _ = EmbeddingModelFactory.create_embedding_model(provider="invalid")

import os

from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from agentblock.tools import load_config
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def test_create_from_yaml_openai():
    """
    정상적인 openai provider 설정이 담긴 YAML을 임시 생성하고,
    create_from_yaml로 제대로 OpenAIEmbeddings 인스턴스를
    얻어오는지 확인한다.
    """
    path_openai_embedding = load_config.get_abspath(
        "../../yaml/embeddings/openai_embedding.yaml", script_dir
    )
    assert os.path.exists(path_openai_embedding)

    # 3) create_from_yaml 호출
    embedding_obj = EmbeddingModelFactory.from_yaml_file(path_openai_embedding)

    # 4) 타입 확인
    assert isinstance(
        embedding_obj, OpenAIEmbeddings
    ), f"Expected OpenAIEmbeddings, got {type(embedding_obj)}"

    # 5) 실제 임베딩 함수 호출
    vector = embedding_obj.embed_query("Hello test")
    assert isinstance(vector, list), "Embedding result must be a list"
    assert len(vector) > 0, "Embedding vector cannot be empty"

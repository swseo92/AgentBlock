import pytest
import tempfile
import os
import yaml

from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


load_dotenv()


def test_create_from_yaml_openai():
    """
    정상적인 openai provider 설정이 담긴 YAML을 임시 생성하고,
    create_from_yaml로 제대로 OpenAIEmbeddings 인스턴스를
    얻어오는지 확인한다.
    """
    # 1) 임시 YAML 작성
    test_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002",
        }
    }

    # 2) 임시 파일에 YAML 저장
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(
            test_config,
            tmp_file,
        )
        tmp_file_path = tmp_file.name

    # 3) create_from_yaml 호출
    embedding_obj = EmbeddingModelFactory.create_from_yaml(tmp_file_path)

    # 4) 타입 확인
    assert isinstance(
        embedding_obj, OpenAIEmbeddings
    ), f"Expected OpenAIEmbeddings, got {type(embedding_obj)}"

    # 5) 실제 임베딩 함수 호출
    vector = embedding_obj.embed_query("Hello test")
    assert isinstance(vector, list), "Embedding result must be a list"
    assert len(vector) > 0, "Embedding vector cannot be empty"

    # 임시 파일 삭제
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)


def test_create_from_yaml_unsupported():
    """
    지원하지 않는 provider가 들어간 YAML로 인스턴스를 생성하면
    ValueError가 발생해야 한다.
    """
    test_config = {"embedding": {"provider": "nonexistent", "model": "some-model"}}

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(test_config, tmp_file)
        tmp_file_path = tmp_file.name

    with pytest.raises(ValueError) as exc_info:
        EmbeddingModelFactory.create_from_yaml(tmp_file_path)

    assert "Unsupported embedding model provider" in str(exc_info.value)

    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)


def test_create_from_yaml_no_embedding_section():
    """
    'embedding' 키가 없는 YAML을 넣었을 때의 동작을 체크한다.
    현재 create_from_yaml 구현에 따라,
    default 값("openai", "text-embedding-ada-002")가 사용되거나,
    or 에러가 날 수도 있으므로, 코드에 맞게 테스트를 조정한다.
    """
    test_config = {"dummy_key": "dummy_value"}

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as tmp_file:
        yaml.dump(test_config, tmp_file)
        tmp_file_path = tmp_file.name

    # create_from_yaml 내부에서 get("embedding", {})로 default {}를 반환하므로,
    # provider="openai", model="text-embedding-ada-002" 가 사용될 것으로 예상
    embedding_obj = EmbeddingModelFactory.create_from_yaml(tmp_file_path)

    assert isinstance(embedding_obj, OpenAIEmbeddings)
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

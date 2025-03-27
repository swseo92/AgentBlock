import pytest
import os

from agentblock.embedding.embedding_factory import EmbeddingModelFactory


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

base_path = f"{script_dir}/error_case/"


def test_create_from_yaml_unsupported():
    """
    지원하지 않는 provider가 들어간 YAML로 인스턴스를 생성하면
    ValueError가 발생해야 한다.
    """

    path = f"{base_path}/unsupported_provider.yaml"
    with pytest.raises(ValueError) as exc_info:
        EmbeddingModelFactory.from_yaml_file(path)

    assert "Unsupported embedding model provider" in str(exc_info.value)

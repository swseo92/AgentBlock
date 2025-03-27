import yaml
from typing import Dict, Any
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from agentblock.embedding.dummy_embedding import DummyEmbedding


class EmbeddingModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_embedding_model(provider: str = "openai", **kwargs) -> Embeddings:
        """
        provider와 model에 따라 Embeddings 인스턴스를 생성한다.
        kwargs로 api_key, dimension 등 추가 파라미터를 받을 수 있다.
        """
        if provider == "openai":
            return OpenAIEmbeddings(**kwargs)
        elif provider == "dummy":
            return DummyEmbedding(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")

    @staticmethod
    def from_yaml(config: Dict[str, Any]) -> Embeddings:
        """
        이미 YAML 로드(safe_load)된 config 딕셔너리를 입력받아,
        해당 embedding 설정을 기반으로 Embeddings 인스턴스를 생성한다.

        예시:
        config = {
          "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002",
            "api_key": "...",
            ...
          }
        }
        """
        embedding_conf = config["config"]

        provider = embedding_conf["provider"]
        # 나머지 파라미터들을 extra_kwargs로
        extra_kwargs = dict(embedding_conf)
        extra_kwargs.pop("provider", None)

        return EmbeddingModelFactory.create_embedding_model(
            provider=provider, **extra_kwargs
        )

    @staticmethod
    def from_yaml_file(yaml_path: str) -> Embeddings:
        """
        YAML 파일 경로를 입력받아 파일을 로드(safe_load) 한 뒤,
        from_yaml 메서드를 이용해 Embeddings 인스턴스를 생성한다.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)["nodes"][0]
        return EmbeddingModelFactory.from_yaml(config)

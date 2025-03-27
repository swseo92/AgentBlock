from typing import Dict, Any
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from agentblock.embedding.dummy_embedding import DummyEmbedding
from agentblock.tools.load_config import get_yaml_for_single_node_file


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

    def from_yaml_file_single_node(self, yaml_path):
        config = get_yaml_for_single_node_file(yaml_path)
        print(config)
        return self.from_yaml(config)


if __name__ == "__main__":
    model = EmbeddingModelFactory().from_yaml_file_single_node(
        "template_embedding.yaml"
    )

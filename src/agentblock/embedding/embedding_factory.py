import yaml
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_embedding_model(
        provider: str = "openai", model: str = "text-embedding-ada-002", **kwargs
    ) -> Embeddings:
        """
        provider와 model_name에 따라 적절한 임베딩 모델 인스턴스를 생성합니다.
        현재는 provider가 "openai"일 경우 OpenAIEmbeddings를 반환하며,
        향후 다른 공급자를 위한 구현체를 추가할 수 있습니다.

        kwargs로 api_key 등 추가 파라미터를 받을 수 있습니다.
        """
        if provider == "openai":
            # 여기서 kwargs.get("openai_api_key") 등을 넣어줄 수도 있음
            return OpenAIEmbeddings(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")

    @staticmethod
    def create_from_yaml(yaml_path: str) -> Embeddings:
        """
        yaml_path에서 embedding 설정을 읽어,
        create_embedding_model(...)을 호출하여 Embeddings 인스턴스를 생성합니다.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(
                f
            )  # 예: {"embedding": {"provider": "openai", "model": "..."}}

        # 1) embedding 섹션 파싱
        embedding_conf = config_data.get("embedding", {})
        provider = embedding_conf.get("provider", "openai")
        model_name = embedding_conf.get("model", "text-embedding-ada-002")

        # 2) 추가 파라미터 (예: api_key 등) 읽어오기
        extra_kwargs = dict(embedding_conf)
        # provider, model 키는 이미 썼으니 제거
        extra_kwargs.pop("provider", None)
        extra_kwargs.pop("model", None)

        # 3) create_embedding_model 호출
        return EmbeddingModelFactory.create_embedding_model(
            provider=provider, model=model_name, **extra_kwargs
        )

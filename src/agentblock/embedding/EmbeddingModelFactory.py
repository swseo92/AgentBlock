from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_embedding_model(
        provider: str = "openai", model: str = "text-embedding-ada-002"
    ) -> Embeddings:
        """
        provider와 model_name에 따라 적절한 임베딩 모델 인스턴스를 생성합니다.
        현재는 provider가 "openai"일 경우 LangchainOpenAIEmbeddingModel을 반환하며,
        향후 다른 공급자를 위한 구현체를 추가할 수 있습니다.
        """
        if provider == "openai":
            return OpenAIEmbeddings(model=model)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")

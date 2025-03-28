from typing import Dict

from agentblock.base import BaseReference
from agentblock.embedding.dummy_embedding import DummyEmbedding
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


class EmbeddingReference(BaseReference):
    """
    LangChain Embeddings 객체를 생성 및 보관하는 비실행 노드.
    build() 호출 시 langchain.embeddings.Embeddings 인스턴스를 반환하고,
    내부에도 저장(_embedding)에 보관할 수 있음.
    """

    def __init__(self, name: str, provider: str, config: Dict = None):
        super().__init__(name)
        self.provider = provider  # 예: "openai", "huggingface"
        self.config = config
        self._embedding = None  # build() 완료 후, langchain Embeddings 객체

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, "BaseReference"]
    ) -> "EmbeddingReference":
        ref_name = config["name"]
        cfg = config.get("config", {})
        provider = cfg.get("provider")

        return EmbeddingReference(
            name=ref_name,
            provider=provider,
            config=cfg,
        )

    def build(self):
        """
        LangChain의 Embeddings 객체를 생성하여 반환.
        build()가 여러 번 호출되어도, 한번 생성하면 _embedding이 캐싱되도록 할 수 있음.
        """
        if self._embedding is not None:
            # 이미 build()가 완료된 상태라면 다시 생성할 필요가 없을 수 있음
            return self._embedding

        param_dict = self.config.get("param", {})
        if self.provider == "openai":
            # openai_api_key가 필요한 경우
            self._embedding = OpenAIEmbeddings(**param_dict)
        elif self.provider == "huggingface":
            # 예: huggingface_hub_api_key, 또는 로컬 모델 경로 등
            self._embedding = HuggingFaceEmbeddings(**param_dict)
        elif self.provider == "dummy":
            self._embedding = DummyEmbedding(**param_dict)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        return self._embedding

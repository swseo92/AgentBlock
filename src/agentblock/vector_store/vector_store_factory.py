import yaml
from langchain_core.embeddings.embeddings import Embeddings
from agentblock.vector_store.faiss_utils import create_faiss_vector_store
import agentblock.tools.load_config as load_config
import os


class VectorStoreFactory:
    """
    path + embedding_model 식별자를 조합해 캐싱하는 예시.
    캐시 키가 일치하면, 이미 로드된 인스턴스를 반환해 메모리 사용을 줄인다.
    """

    _cache = {}  # (path, model_identifier) -> FAISS object

    def __init__(self):
        pass

    def create_vector_store(
        self, provider: str = "faiss", embedding_model: Embeddings = None, **kwargs
    ):
        """
        provider와 embedding_model에 따라 적절한 벡터 스토어 인스턴스를 생성합니다.
        현재는 provider가 "faiss"일 경우 LangchainFAISSVectorStore,
        "annoy"일 경우 AnnoyVectorStore를 반환합니다.
        """
        if provider == "faiss":
            if embedding_model is None:
                raise ValueError(
                    "embedding_model must be provided for FAISS vector store."
                )
            return create_faiss_vector_store(embedding_model, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    def from_yaml(self, yaml_path: str, embedding_model: Embeddings):
        """
        YAML에서 vector_store 설정을 읽어와,
        path + model 식별자를 키로 해 캐싱 로직을 적용.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        vector_section = config_data.get("vector_store", {})
        provider = vector_section.get("provider", "faiss")
        vs_config = vector_section.get("config", {})

        path = vs_config.get("path", None)
        if path is not None:
            base_path = load_config.get_parent_dir_abspath(yaml_path)
            path = load_config.get_abspath(path, base_path)
            assert os.path.exists(path)
            vs_config.update({"path": path})

        # 모델 식별자 획득 (e.g. model_name, dimension)
        model_identifier = self._get_model_identifier(embedding_model)

        # 캐싱 키
        cache_key = (path, model_identifier)

        # 이미 같은 path + model_identifier로 로드했다면, 캐시 재사용
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 새로 로드
        vs_obj = self.create_vector_store(provider, embedding_model, **vs_config)

        # 캐시에 저장
        self._cache[cache_key] = vs_obj
        return vs_obj

    def _get_model_identifier(self, embedding_model: Embeddings) -> str:
        # 예: OpenAIEmbeddings에 model 속성이 있다면:
        # return getattr(embedding_model, "model", str(embedding_model))

        # 여기서는 그냥 임시로 str() 변환
        return str(embedding_model)

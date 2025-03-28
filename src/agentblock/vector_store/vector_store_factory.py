from langchain_core.embeddings.embeddings import Embeddings
from agentblock.vector_store.faiss_utils import create_faiss_vector_store
from agentblock.tools.load_config import (
    get_yaml_for_single_node_file,
    get_parent_dir_abspath,
    get_abspath,
)


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

    def from_yaml(self, config: str, embedding_model: Embeddings, base_dir=None):
        assert base_dir is not None, "base_dir must be provided"

        vs_config = config["config"]

        provider = vs_config["provider"]
        path = vs_config.get("path", None)
        if path is not None:
            path = get_abspath(path, base_dir)
            vs_config.update({"path": path})

        # 모델 식별자 획득 (e.g. model_name, dimension)
        model_identifier = self._get_model_identifier(embedding_model)

        # 캐싱 키
        cache_key = (path, model_identifier)

        # 이미 같은 path + model_identifier로 로드했다면, 캐시 재사용
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 새로 로드
        vs_obj = self.create_vector_store(provider, embedding_model, path=path)

        # 캐시에 저장
        self._cache[cache_key] = vs_obj
        return vs_obj

    def _get_model_identifier(self, embedding_model: Embeddings) -> str:
        # 예: OpenAIEmbeddings에 model 속성이 있다면:
        # return getattr(embedding_model, "model", str(embedding_model))

        # 여기서는 그냥 임시로 str() 변환
        return str(embedding_model)

    def from_yaml_file_single_node(self, yaml_path, embedding_model: Embeddings):
        config = get_yaml_for_single_node_file(yaml_path)
        base_dir = get_parent_dir_abspath(yaml_path)
        vector_store = self.from_yaml(config, embedding_model, base_dir=base_dir)
        return vector_store


if __name__ == "__main__":
    from agentblock.embedding.dummy_embedding import DummyEmbedding
    from agentblock.schema.tools import validate_yaml

    path_yaml = "template_vector_store.yaml"
    validate_yaml(path_yaml)

    model = DummyEmbedding()
    factory = VectorStoreFactory()
    vs = factory.from_yaml_file_single_node(path_yaml, model)

from agentblock.base import BaseNode
from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from agentblock.vector_store.vector_store_factory import VectorStoreFactory
import agentblock.tools.load_config as load_config


class RetrieverNode(BaseNode):
    def __init__(
        self,
        name: str,
        input_keys: list,
        output_key: str,
        retriever_config: dict,
        base_dir: str = None,
    ):
        """
        retriever_config 예시 (모든 필드가 필수):
        {
          "embedding": {"from_file": "embedding.yaml"},  # or "name": "...", ...
          "vector_store": {"from_file": "vs.yaml"},      # or "name": "...", ...
          "search_method": "invoke",
          "search_type": "similarity",
          "search_kwargs": {"k": 5}
        }
        """
        super().__init__(name)
        self.input_keys = input_keys
        self.output_key = output_key
        self.retriever_conf = retriever_config
        self.base_dir = base_dir

        self.vector_store = None

    @staticmethod
    def from_yaml(config: dict, base_dir: str) -> "RetrieverNode":
        """
        config는 이미 yaml.safe_load()된 dict (예: template_retriever.yaml).
        'type': 'retriever' 노드를 찾아 RetrieverNode 인스턴스를 생성한다.
        """
        # 필수 필드 직접 참조
        if "name" not in config:
            raise ValueError("retriever 노드에 'name'이 없습니다.")
        name = config["name"]

        if "input_keys" not in config:
            raise ValueError(f"retriever 노드 '{name}'에 'input_keys'가 없습니다.")
        input_keys = config["input_keys"]

        if "output_key" not in config:
            raise ValueError(f"retriever 노드 '{name}'에 'output_key'가 없습니다.")
        output_key = config["output_key"]

        if "config" not in config:
            raise ValueError(f"retriever 노드 '{name}'에 'config' 필드가 없습니다.")
        retriever_config = config["config"]

        return RetrieverNode(
            name=name,
            input_keys=input_keys,
            output_key=output_key,
            retriever_config=retriever_config,
            base_dir=base_dir,
        )

    def build(self):
        """
        빌드 단계:
          1) embedding 설정 로드
          2) vector_store 설정 로드
          3) search_type / search_kwargs / search_method 확인 (모두 필수!)
          4) 최종 node_fn 반환
        """
        # 1) 임베딩
        if "embedding" not in self.retriever_conf:
            raise ValueError("retriever_config에 'embedding'이 없습니다.")
        embedding_conf = self.retriever_conf["embedding"]
        embedding_model = _load_embedding(embedding_conf, self.base_dir)

        # 2) 벡터스토어
        if "vector_store" not in self.retriever_conf:
            raise ValueError("retriever_config에 'vector_store'가 없습니다.")
        vector_store_conf = self.retriever_conf["vector_store"]
        self.vector_store = _load_vector_store(
            vector_store_conf, embedding_model, self.base_dir
        )

        # 3) search_method / search_type / search_kwargs (모두 필수!)
        if "search_method" not in self.retriever_conf:
            raise ValueError("retriever_config에 'search_method'가 없습니다.")
        search_method = self.retriever_conf["search_method"]

        if "search_type" not in self.retriever_conf:
            raise ValueError("retriever_config에 'search_type'가 없습니다.")
        search_type = self.retriever_conf["search_type"]

        if "search_kwargs" not in self.retriever_conf:
            raise ValueError("retriever_config에 'search_kwargs'가 없습니다.")
        search_kwargs = self.retriever_conf["search_kwargs"]

        retriever = self.vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        def node_fn(state):
            # input_keys가 여러 개인 경우, 여기서는 첫 번째만 사용
            if not self.input_keys:
                raise ValueError(f"RetrieverNode '{self.name}'에 input_keys가 비어있습니다.")
            query_key = self.input_keys[0]

            if query_key not in state:
                raise ValueError(f"state에 '{query_key}' 키가 없습니다.")
            query_val = state[query_key]

            # retriever에서 search_method 호출
            search_fn = getattr(retriever, search_method, None)
            if not search_fn:
                raise ValueError(f"Retriever does not have method '{search_method}'")

            results = search_fn(query_val)
            return {self.output_key: results}

        return node_fn


def _load_embedding(emb_conf: dict, base_dir: str):
    """
    embedding:
      from_file: "embedding.yaml"   # 실제 파일
      # or name: "dummy_embedding"  # (미구현: NotImplementedError)
      # or provider/dimension ...
    """
    # name + from_file 동시 지정 시 에러
    if "name" in emb_conf and "from_file" in emb_conf:
        raise ValueError("Cannot specify both 'name' and 'from_file'")

    # from_file 분기
    if "from_file" in emb_conf:
        path = load_config.get_abspath(emb_conf["from_file"], base_dir)
        return EmbeddingModelFactory().from_yaml_file_single_node(path)

    # name 분기 (미구현)
    elif "name" in emb_conf:
        raise NotImplementedError(
            "'name' reference for embedding is not implemented in this example."
        )

    # 직접 provider/dimension ...
    else:
        return EmbeddingModelFactory.from_yaml({"embedding": emb_conf})


def _load_vector_store(vs_conf: dict, embedding_model, base_dir: str):
    """
    vector_store:
      from_file: "vs.yaml"
      # or
      name: "my_faiss"
    """
    if "from_file" in vs_conf:
        path = load_config.get_abspath(vs_conf["from_file"], base_dir)

        return VectorStoreFactory().from_yaml_file_single_node(
            yaml_path=path, embedding_model=embedding_model
        )
    elif "name" in vs_conf:
        raise NotImplementedError(
            "'name' reference for vector_store is not implemented in this example."
        )
    else:
        return VectorStoreFactory().from_yaml(vs_conf, embedding_model=embedding_model)

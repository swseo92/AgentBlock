import yaml
from agentblock.base import BaseNode
from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from agentblock.vector_store.vector_store_factory import VectorStoreFactory
import agentblock.tools.load_config as load_config


class RetrieverNode(BaseNode):
    def __init__(
        self,
        name: str,
        embedding_path: str,
        vector_store_path: str,
        retriever_conf: dict,
        input_keys: list,
        output_key: str,
    ):
        """
        embedding_path: 임베딩 설정 YAML 파일 경로
        vector_store_path: 벡터스토어 설정 YAML 파일 경로
        retriever_conf: 검색 관련 파라미터 (search_method, search_type, search_kwargs 등)
        input_keys: 노드가 입력으로 받을 state 키 (예: ["query"])
        output_key: 노드가 최종적으로 반환할 state 키 (예: "retrieved_docs")
        """
        super().__init__(name)
        self.embedding_path = embedding_path
        self.vector_store_path = vector_store_path
        self.retriever_conf = retriever_conf
        self.input_keys = input_keys
        self.output_key = output_key

    @staticmethod
    def from_yaml(yaml_path: str) -> "RetrieverNode":
        """
        yaml_path로부터 전체 YAML 파일을 로드하고,
        "type": "retriever" 노드를 찾은 뒤 RetrieverNode 인스턴스를 생성.

        YAML 예시:
          nodes:
            - name: my_retriever
              type: retriever
              embedding_path: "elements/embedding.yaml"
              vector_store_path: "elements/vector_store.yaml"
              retriever_config:
                search_method: invoke
                search_type: similarity
                search_kwargs:
                  k: 5
              input_keys:
                - query
              output_key: retrieved_docs
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        nodes_data = data.get("nodes", [])
        retriever_node_data = None

        # 여러 노드 중 "type": "retriever" 인 것만 찾음
        for nd in nodes_data:
            if nd.get("type") == "retriever":
                retriever_node_data = nd
                break

        if not retriever_node_data:
            raise ValueError("No node with 'type: retriever' found in the YAML.")

        name = retriever_node_data.get("name", "unnamed_retriever")
        embedding_path = retriever_node_data["embedding_path"]
        vector_store_path = retriever_node_data["vector_store_path"]
        retriever_conf = retriever_node_data.get("retriever_config", {})
        input_keys = retriever_node_data.get("input_keys", [])
        output_key = retriever_node_data.get("output_key", "retrieved_docs")

        # 상대 경로로 저장된 path를 절대경로로 치환
        base_path = load_config.get_parent_dir_abspath(yaml_path)
        embedding_path = load_config.get_abspath(embedding_path, base_path)
        vector_store_path = load_config.get_abspath(vector_store_path, base_path)

        return RetrieverNode(
            name=name,
            embedding_path=embedding_path,
            vector_store_path=vector_store_path,
            retriever_conf=retriever_conf,
            input_keys=input_keys,
            output_key=output_key,
        )

    def build(self):
        """
        빌드 단계:
        1) embedding_path를 이용해 임베딩 생성
        2) vector_store_path를 이용해 벡터스토어 생성
        3) retriever 객체(as_retriever) 생성
        4) search_method에 따라 동적 메서드 호출
        """
        # 1) 임베딩 생성
        embedding_model = EmbeddingModelFactory.create_from_yaml(self.embedding_path)

        # 2) 벡터 스토어 생성
        vector_store = VectorStoreFactory().create_from_yaml(
            yaml_path=self.vector_store_path, embedding_model=embedding_model
        )

        # 3) langchain Retriever
        search_type = self.retriever_conf.get("search_type", "similarity")
        search_kwargs = self.retriever_conf.get("search_kwargs", {})
        retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        # 4) search_method (예: "invoke", "get_relevant_documents" 등)
        search_method = self.retriever_conf.get("search_method", "invoke")

        def node_fn(state):
            # 상태에서 query 추출
            query = state.get("query", "")

            # retriever에 search_method 이름의 메서드가 있는지 확인
            search_fn = getattr(retriever, search_method, None)
            if not search_fn:
                raise ValueError(f"Retriever does not have method '{search_method}'")

            # 검색 실행
            results = search_fn(query)

            # 반환
            return {self.output_key: results}

        return node_fn

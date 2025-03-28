from typing import Dict, Any
from agentblock.base import BaseNode


class RetrieverNode(BaseNode):
    """
    새 스키마용 Retriever 노드.
    - BFS 실행 시, state["query"]를 받아 vector_store에서 검색
    - 구버전에서 가져온 search_method, search_type, search_kwargs 로직 반영
    """

    def __init__(
        self,
        name: str,
        input_keys: list,
        output_key: str,
        vector_store: Any = None,
        search_method: str = "invoke",
        search_type: str = "similarity",
        search_kwargs: dict = None,
    ):
        super().__init__(name)
        self.input_keys = input_keys
        self.output_key = output_key

        # 이미 build()된 LangChain VectorStore (예: FAISS, Chroma 등)
        self.vector_store = vector_store

        # 구버전에서 쓰던 검색 파라미터
        self.search_method = search_method
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, Any]
    ) -> "RetrieverNode":
        """
        새 스키마:
          config["config"]["reference"]["vector_store"] 에서 vector_store 이름을 찾고,
          references_map[that_name]에서 실제 VectorStore 객체 획득.
          search_method / search_type / search_kwargs도 함께 파싱.
        """
        node_name = config["name"]
        input_keys = config["input_keys"]
        output_key = config["output_key"]

        node_cfg = config.get("config", {})
        # 구버전 로직 통합
        search_method = node_cfg.get("search_method", "invoke")
        search_type = node_cfg.get("search_type", "similarity")
        search_kwargs = node_cfg.get("search_kwargs", {})

        # vector_store 참조
        ref_links = node_cfg.get("reference", {})
        vs_name = ref_links.get("vector_store")
        if not vs_name:
            raise ValueError(f"RetrieverNode '{node_name}'에 vector_store 참조가 없습니다.")
        vector_store_obj = references_map.get(vs_name)
        if not vector_store_obj:
            raise ValueError(
                f"VectorStore '{vs_name}' not found in references_map for retriever '{node_name}'"
            )

        return RetrieverNode(
            name=node_name,
            input_keys=input_keys,
            output_key=output_key,
            vector_store=vector_store_obj,
            search_method=search_method,
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def build(self):
        """
        BFS에서 이 Node가 실행될 때 호출될 함수(node_fn)를 반환.
        node_fn이 query를 받아 vector_store 검색, 결과를 state에 저장.
        """

        def node_fn(state: Dict) -> Dict:
            # 1) query 가져오기
            inputs = self.get_inputs(state)
            if not self.input_keys:
                raise ValueError(f"RetrieverNode '{self.name}'에 input_keys가 비어있습니다.")
            query_key = self.input_keys[0]

            if query_key not in inputs:
                raise ValueError(
                    f"state에 '{query_key}' 키가 없습니다 (RetrieverNode '{self.name}')."
                )
            query_val = inputs[query_key]

            # 2) vector_store.as_retriever()로 검색객체 생성
            #    or 직접 similarity_search 호출
            # 여기서는 구버전 방식( as_retriever + getattr(retriever, search_method) )을 예시
            if not hasattr(self.vector_store, "as_retriever"):
                raise TypeError("vector_store 객체가 'as_retriever()' 메서드를 지원하지 않습니다.")

            retriever = self.vector_store.as_retriever(
                search_type=self.search_type, search_kwargs=self.search_kwargs
            )
            search_fn = getattr(retriever, self.search_method, None)
            if not search_fn:
                raise ValueError(f"retriever에 메서드 '{self.search_method}'가 없습니다.")

            # 3) 검색 수행
            results = search_fn(query_val)

            # 4) 결과를 BFS state에 저장
            return {self.output_key: results}

        return node_fn

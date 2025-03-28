import os
import yaml
from typing import TypedDict, Type, Set, Any

from langgraph.graph import StateGraph, START, END
from agentblock.llm.llm_node import LLMNode
from agentblock.function import FunctionFromFileNode, FunctionFromLibraryNode
from agentblock.retriever.retriever_node import RetrieverNode
from agentblock.schema.tools import validate_yaml


# 노드 타입 매핑
NODE_TYPE_MAP = {
    "llm": LLMNode,
    "function_from_file": FunctionFromFileNode,
    "function_from_library": FunctionFromLibraryNode,
    "from_yaml": "handled separately",  # 따로 처리
    "retriever": RetrieverNode,
}


class GraphBuilder:
    def __init__(self, path: str):
        """
        path_or_dict가 문자열(파일 경로)이면:
          - 절대 경로 계산 후, YAML을 로드하고
          - self.yaml_dir를 그 파일의 디렉토리로 설정
        dict로 직접 주어졌다면:
          - self.yaml_dir = None
        """
        self.yaml_path = os.path.abspath(path)
        self.validate_yaml()

        self.yaml_dir = os.path.dirname(self.yaml_path)
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 노드 및 엣지 정의
        self.node_defs = self.config["nodes"]
        self.edge_defs = self.config.get("edges", [])
        self.node_map = {}  # name → node_fn
        self.used_keys: Set[Any] = set()

    def validate_yaml(self):
        validate_yaml(self.yaml_path)

    def generate_state(self) -> Type[TypedDict]:
        """
        used_keys를 바탕으로 TypedDict 타입을 동적으로 생성
        """
        state_dict = {}
        for k in self.used_keys:
            state_dict[k] = Any
        return TypedDict("State", state_dict, total=False)

    def load_nodes(self):
        """
        노드별로 실제 객체(LLM, Function, 서브그래프)를 생성.
        """
        for node_cfg in self.node_defs:
            node_type = node_cfg["type"]

            if node_type == "from_yaml":
                from_file = node_cfg["config"].get("from_file")
                # 현재 yaml_dir를 기준으로 경로 조합
                sub_file_path = os.path.join(self.yaml_dir, from_file)

                # 재귀적으로 서브그래프 생성
                sub_builder = GraphBuilder(sub_file_path)
                sub_graph = sub_builder.build_graph()

                self.node_map[node_cfg["name"]] = sub_graph
                # 서브그래프 사용 키도 상위 그래프에 합침
                self.used_keys.update(sub_builder.used_keys)

            else:
                # llm, function, etc.
                cls = NODE_TYPE_MAP.get(node_type)
                if not cls:
                    raise ValueError(f"Unsupported node type: {node_type}")
                node_obj = cls.from_yaml(node_cfg, base_dir=self.yaml_dir)

                node_fn = node_obj.build()
                self.node_map[node_cfg["name"]] = node_fn

                # input_keys / output_key → used_keys에 추가
                self.used_keys.update(node_cfg.get("input_keys", []))
                out_key = node_cfg.get("output_key")
                if isinstance(out_key, str):
                    self.used_keys.add(out_key)
                elif isinstance(out_key, list):
                    for k in out_key:
                        self.used_keys.add(k)

    def build_graph(self):
        """
        노드를 로딩한 뒤 LangGraph StateGraph를 구성하고,
        START/END 예약어를 치환해 컴파일된 그래프를 반환.
        """
        # 1) 노드 로딩
        self.load_nodes()

        # 2) State TypedDict 생성
        State = self.generate_state()
        graph = StateGraph(State)

        # 3) 그래프에 노드 등록
        for name, fn in self.node_map.items():
            graph.add_node(name, fn)

        # 4) 에지 연결
        for edge in self.edge_defs:
            from_name = edge["from"]
            to_name = edge["to"]

            # START/END 치환
            if from_name == "START":
                from_name = START
            if to_name == "END":
                to_name = END

            # condition 분기 처리 vs 일반 edge
            if "condition" in edge:
                # 이 예시에서는 route 키를 써서 condition 매칭
                graph.add_conditional_edges(
                    from_name,
                    lambda state: state.get("route"),
                    {edge["condition"]: to_name},
                )
            else:
                graph.add_edge(from_name, to_name)

        return graph.compile()

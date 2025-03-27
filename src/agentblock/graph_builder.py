import os
import yaml
from typing import TypedDict, Type, Set, Any
from collections import defaultdict, deque

from langgraph.graph import StateGraph, START, END
from agentblock.llm.llm_node import LLMNode
from agentblock.function.function_node import FunctionNode

# 노드 타입 매핑
NODE_TYPE_MAP = {
    "llm": LLMNode,
    "function": FunctionNode,
    "from_yaml": "handled separately",  # 따로 처리
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
        self.yaml_dir = os.path.dirname(self.yaml_path)
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 간단한 스키마 검증
        self.validate_config()

        # 노드 및 엣지 정의
        self.node_defs = self.config["nodes"]
        self.edge_defs = self.config.get("edges", [])
        self.node_map = {}  # name → node_fn
        self.used_keys: Set[Any] = set()

    def validate_config(self):
        """
        1) config가 dict인지
        2) nodes 필드가 있는지
        3) node/edge 구조 검사
        """
        if not isinstance(self.config, dict):
            raise ValueError("Graph config must be a dict.")

        # nodes 필드는 필수
        if "nodes" not in self.config:
            raise ValueError("Graph config must have 'nodes' field.")

        # 노드, 에지 검증
        validate_nodes(self.config)
        self.validate_edges()

    def validate_edges(self):
        """
        edge 구조 검사 + 단절 노드 검사
        """
        validate_edges(self.config)
        validate_no_disconnected_nodes(self.config)

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
                # 서브 YAML 로딩
                from_file = node_cfg.get("from_file")
                if from_file:
                    # 현재 yaml_dir를 기준으로 경로 조합
                    if self.yaml_dir:
                        sub_file_path = os.path.join(self.yaml_dir, from_file)
                    else:
                        # yaml_dir가 없으면 그대로
                        sub_file_path = from_file

                    # 재귀적으로 서브그래프 생성
                    sub_builder = GraphBuilder(sub_file_path)
                    sub_graph = sub_builder.build_graph()

                    self.node_map[node_cfg["name"]] = sub_graph
                    # 서브그래프 사용 키도 상위 그래프에 합침
                    self.used_keys.update(sub_builder.used_keys)

                # inline graph 방식이면 node_cfg["graph"]를 그대로 빌드
                elif "graph" in node_cfg:
                    sub_config = node_cfg["graph"]
                    sub_builder = GraphBuilder(sub_config)
                    sub_graph = sub_builder.build_graph()
                    self.node_map[node_cfg["name"]] = sub_graph
                    self.used_keys.update(sub_builder.used_keys)
                else:
                    raise ValueError("from_yaml node must have 'from_file' or 'graph'.")

            else:
                # llm, function, etc.
                cls = NODE_TYPE_MAP.get(node_type)
                if not cls:
                    raise ValueError(f"Unsupported node type: {node_type}")
                if node_type == "function":
                    node_obj = cls.from_yaml(node_cfg, base_dir=self.yaml_dir)
                else:
                    node_obj = cls.from_yaml(node_cfg)

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


def validate_nodes(config):
    """
    노드 검증:
    - node는 dict
    - name, type 필수
    - from_yaml인 경우 from_file or graph 필요
    - 그 외(llm, function 등)는 input_keys, output_key 필요
    """
    nodes = config["nodes"]
    for idx, node_cfg in enumerate(nodes):
        if not isinstance(node_cfg, dict):
            raise ValueError(f"Node at index {idx} must be a dict.")

        if "name" not in node_cfg or "type" not in node_cfg:
            raise ValueError(f"Node at index {idx} must have 'name' and 'type'.")

        node_type = node_cfg["type"]
        if node_type == "from_yaml":
            if "from_file" not in node_cfg and "graph" not in node_cfg:
                raise ValueError("from_yaml node must have 'from_file' or 'graph'.")

        else:
            # llm, function, etc.
            required_fields = ["input_keys", "output_key"]
            for rf in required_fields:
                if rf not in node_cfg:
                    raise ValueError(
                        f"Graph config node of type '{node_type}' must have '{rf}'."
                    )


def validate_edges(config):
    """
    엣지 검증:
    - edges 필드가 list인지
    - 각 edge에 from, to 존재
    - START로 시작하는 edge >= 1, END로 끝나는 edge >= 1
    - END로 가는 edge가 여러 개인 경우 에러
    """
    edges = config.get("edges", None)
    if not edges or not isinstance(edges, list):
        raise ValueError("Missing or invalid 'edges'. Must be a non-empty list.")

    has_edge_from_start = False
    has_edge_to_end = False
    list_edge_to_end = []

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge at index {i} must be a dict.")
        if "from" not in edge:
            raise ValueError(f"Edge at index {i} is missing 'from'.")
        if "to" not in edge:
            raise ValueError(f"Edge at index {i} is missing 'to'.")

        if edge["from"] == "START":
            has_edge_from_start = True
        if edge["to"] == "END":
            list_edge_to_end.append(edge)
            has_edge_to_end = True
            if len(list_edge_to_end) > 1:
                raise ValueError("More than one edge leads to 'END'.")

    if not has_edge_from_start:
        raise ValueError("No edge starts from 'START'.")
    if not has_edge_to_end:
        raise ValueError("No edge leads to 'END'.")


def validate_no_disconnected_nodes(config):
    """
    단절 노드 검사:
    - START에서 도달 불가능한 노드
    - END로 도달 불가능한 노드
    BFS(정방향, 역방향)로 체크
    """
    nodes = config.get("nodes", [])
    edges = config.get("edges", [])

    node_names = {n["name"] for n in nodes}

    # 정방향 인접 리스트
    fwd_adj = defaultdict(list)
    for e in edges:
        fwd_adj[e["from"]].append(e["to"])

    # 역방향 인접 리스트
    rev_adj = defaultdict(list)
    for e in edges:
        rev_adj[e["to"]].append(e["from"])

    # 1) START -> ... BFS
    visited_from_start = set()
    queue = deque(["START"])
    while queue:
        current = queue.popleft()
        for nxt in fwd_adj[current]:
            if nxt not in visited_from_start and nxt != "END":
                visited_from_start.add(nxt)
                queue.append(nxt)

    # 2) END -> ... 역방향 BFS
    visited_from_end = set()
    queue = deque(["END"])
    while queue:
        current = queue.popleft()
        for prev in rev_adj[current]:
            if prev not in visited_from_end and prev != "START":
                visited_from_end.add(prev)
                queue.append(prev)

    unreachable_from_start = node_names - visited_from_start
    cannot_reach_end = node_names - visited_from_end

    if unreachable_from_start:
        raise ValueError(
            f"Some nodes are unreachable from START: {unreachable_from_start}"
        )
    if cannot_reach_end:
        raise ValueError(f"Some nodes cannot reach END: {cannot_reach_end}")

import yaml
import collections
from typing import Dict, List, Any, Set, Tuple

EXECUTION_TYPES = {
    "llm",
    "function_from_file",
    "function_from_library",
    "retriever",
    "router",
    "from_yaml",
}
NON_EXECUTION_TYPES = {"embedding", "vector_store"}


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    yaml_path 파일을 로드해 파싱한 dict를 반환한다.
    최상위가 dict 형태인지 간단히 확인한다.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML '{yaml_path}'의 최상위 구조가 dict가 아닙니다.")
    return data


def validate_nodes(
    nodes: List[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
    """
    nodes 배열에 대해:
      - 각 항목이 dict인지
      - 'name', 'type' 필드가 유효한지
      - 노드 이름 중복 여부
      - 실행 노드 vs. 비실행 노드 분류
    반환값: (execution_nodes, non_execution_nodes, node_names)
    """
    node_names: Set[str] = set()
    execution_nodes: List[Dict[str, Any]] = []
    non_execution_nodes: List[Dict[str, Any]] = []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}]가 dict 형태가 아닙니다.")
        name = node.get("name")
        ntype = node.get("type")
        if ntype == "from_yaml":
            try:
                validate_from_yaml_node(node)
            except ValueError as e:
                raise ValueError(f"{name}: {e}")

        if not name or not isinstance(name, str):
            raise ValueError(f"{name}: nodes[{i}]에 'name'이 없거나 문자열이 아닙니다.")
        if name in node_names:
            raise ValueError(f"노드 이름이 중복되었습니다: '{name}'")
        node_names.add(name)

        if not ntype or not isinstance(ntype, str):
            raise ValueError(f"{name}: nodes[{i}]에 'type'이 없거나 문자열이 아닙니다.")

        if ntype in EXECUTION_TYPES:
            execution_nodes.append(node)
        elif ntype in NON_EXECUTION_TYPES:
            non_execution_nodes.append(node)
        else:
            raise ValueError(
                f"지원하지 않는 노드 유형: '{ntype}'. "
                f"허용되는 type: {EXECUTION_TYPES | NON_EXECUTION_TYPES}"
            )

    return execution_nodes, non_execution_nodes, node_names


def validate_edges(edges: List[Any], node_names: Set[str]) -> Tuple[bool, bool]:
    """
    edges 배열에 대해:
      - 각 edge가 dict인지
      - from/to가 문자열인지
      - from=START / to=END / 노드명 존재 여부
    반환값: (has_start_edge, has_end_edge)
    """
    has_start_edge = False
    has_end_edge = False

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"edges[{i}]가 dict 형태가 아닙니다.")
        fr = edge.get("from")
        to = edge.get("to")
        if not isinstance(fr, str) or not isinstance(to, str):
            raise ValueError(f"edges[{i}]에 'from'/'to'가 없거나 문자열이 아닙니다.")

        if fr == "START":
            has_start_edge = True
        else:
            if fr not in node_names:
                raise ValueError(f"edges[{i}]의 from='{fr}'가 유효한 노드명도, START도 아닙니다.")

        if to == "END":
            has_end_edge = True
        else:
            if to not in node_names:
                raise ValueError(f"edges[{i}]의 to='{to}'가 유효한 노드명도, END도 아닙니다.")

    return has_start_edge, has_end_edge


def validate_bfs_for_execution_nodes(
    execution_nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    node_names: Set[str],
) -> None:
    """
    실행 노드가 모두 START→...→END로 이어지는 경로에 포함되는지 BFS로 검사.
    비실행 노드는 제외.
    - START에서 접근 가능한 실행 노드
    - END까지 도달 가능한지
    - 어떤 실행 노드도 누락되지 않았는지
    """
    execution_node_names = {n["name"] for n in execution_nodes}

    # 인접 리스트 구성
    adjacency = {}
    for name in node_names:
        adjacency[name] = []
    # 특별히 "START" 도 인접 목록을 갖도록
    adjacency["START"] = []

    for edge in edges:
        fr = edge["from"]
        to = edge["to"]
        if fr == "START":
            if to in execution_node_names:
                adjacency["START"].append(to)
        elif to == "END":
            # END 자체는 인접 리스트에 넣지 않아도 됨
            if fr in execution_node_names:
                adjacency.setdefault(fr, []).append("END")
        else:
            # fr->to 둘 다 실행 노드면 연결
            if fr in execution_node_names and to in execution_node_names:
                adjacency[fr].append(to)

    visited = set()
    queue = collections.deque()

    # START에서 출발
    for nxt in adjacency["START"]:
        visited.add(nxt)
        queue.append(nxt)

    end_visited = False
    while queue:
        cur = queue.popleft()
        if cur == "END":
            end_visited = True
            continue
        for nxt in adjacency[cur]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                if nxt == "END":
                    end_visited = True

    # 실행 노드 전부 방문됐는지
    missing = execution_node_names - visited
    if missing:
        raise ValueError(f"다음 실행 노드들이 START→...→END 경로에 연결되지 않았습니다: {missing}")

    if not end_visited:
        raise ValueError("BFS 결과, 실행 노드에서 END로 이어지는 경로가 없습니다.")


def check_multiple_nodes_and_single_end(nodes: list, edges: list):
    """
    1) 노드가 여러 개 있는지 (즉 최소 2개 이상인지) 확인
    2) END로 향하는 edge가 정확히 하나만 존재하는지 확인
       - 예: edges 중 to=END인 edge가 여러 개이면 에러
       - 없으면 에러
    """
    # 1) 노드 개수 확인
    if len(nodes) == 0:
        raise ValueError("노드가 2개 이상이어야 합니다. (START, 실행노드, END 등)")

    # 2) END로 향하는 edge 개수
    end_count = sum(1 for e in edges if e.get("to") == "END")
    if end_count == 0:
        raise ValueError("END로 향하는 edge가 하나도 없습니다.")
    if end_count > 1:
        raise ValueError(f"END로 향하는 edge가 {end_count}개 존재합니다. 정확히 하나여야 합니다.")


def check_all_nodes_reach_end(execution_nodes, edges, node_names):
    """
    모든 실행 노드가 END까지 갈 수 있는지 검사.
    BFS or DFS로 각 노드에서 'END'에 도달 가능한지 확인.
    """
    exec_names = {n["name"] for n in execution_nodes}

    # 인접 리스트 (node -> [다음 노드/END])
    adjacency = {name: [] for name in node_names}
    for edge in edges:
        fr = edge["from"]
        to = edge["to"]
        if fr in exec_names:
            adjacency[fr].append(to)

    def can_reach_end(start: str) -> bool:
        visited = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == "END":
                return True
            for nxt in adjacency.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return False

    for node in execution_nodes:
        nname = node["name"]
        if not can_reach_end(nname):
            raise ValueError(f"노드 '{nname}'은(는) END로 이어지는 경로가 없습니다.")


def validate_from_yaml_node(node: dict):
    """
    'type': 'from_yaml' 노드에서,
    반드시 'from_file' 필드가 존재해야 한다.
    """
    if "config" not in node:
        raise ValueError("from_yaml node must have a config field.")
    cfg = node["config"]

    if "from_file" not in cfg:
        raise ValueError("from_yaml node must have 'from_file'")


def validate_yaml(yaml_path: str) -> None:
    data = load_yaml(yaml_path)

    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, list):
        raise ValueError("'nodes' 필드가 없거나 list 형태가 아닙니다.", yaml_path)

    # edges가 없으면 빈 리스트로 처리
    if edges is None:
        edges = []
    print(type(edges))
    if not isinstance(edges, list):
        raise ValueError("'edges' 필드가 list 형태가 아닙니다.", yaml_path)

    # 노드 검증
    try:
        execution_nodes, non_execution_nodes, node_names = validate_nodes(nodes)
    except ValueError as e:
        raise ValueError(f"[validate_yaml] {yaml_path}: {e}")

    # 만약 실행 노드가 하나도 없다면
    if len(execution_nodes) == 0:
        # edges가 비어있어도 OK
        # 비실행 노드만 있음 -> 추가 검증 없이 통과
        # print(f"[validate_yaml] {yaml_path}: 실행 노드 없이 비실행 노드만 정의된 YAML입니다. 통과.")
        return

    # 실행 노드가 있다면, 기존처럼 edges와 BFS 검사 수행
    has_start_edge, has_end_edge = validate_edges(edges, node_names)
    if not has_start_edge:
        raise ValueError("그래프에 START에서 시작하는 edge가 없습니다.")
    if not has_end_edge:
        raise ValueError("그래프에 END로 가는 edge가 없습니다.")

    validate_bfs_for_execution_nodes(execution_nodes, edges, node_names)
    # print(f"[validate_yaml] {yaml_path} 유효성 검사를 통과했습니다.")
    check_multiple_nodes_and_single_end(nodes, edges)
    check_all_nodes_reach_end(execution_nodes, edges, node_names)

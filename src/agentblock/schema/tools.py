import yaml
import collections
from typing import Dict, List, Any, Set, Tuple

# 실행 노드와 비실행 노드(embedding, vector_store 등) 타입을 정의
EXECUTION_TYPES = {
    "llm",
    "function_from_file",
    "function_from_library",
    "retriever",
    "router",
    "from_yaml",
    "data_loader",
    "function",  # 만약 'type: function'을 쓴다면 여기 추가
}
NON_EXECUTION_TYPES = {
    "embedding",
    "vector_store",
    # 필요하다면 "tokenizer", "pdf_loader" 등도 여기 추가 가능
}


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


def validate_top_level_structure(data: dict) -> None:
    """
    data는 YAML을 로드한 후의 최상위 dict.
    - references, nodes, edges 이외의 필드가 있으면 에러
    - references, nodes, edges 중 누락된 필드도 에러(선택적)
      (프로젝트에서 references/nodes/edges가 필수인지 여부에 따라 다르게 처리)
    """

    # 1) 허용되는 필드 지정
    allowed_keys = {"references", "nodes", "edges"}

    # 2) 실제 필드 set
    actual_keys = set(data.keys())

    # 3) 불필요한 필드 검사
    extra_keys = actual_keys - allowed_keys
    if extra_keys:
        raise ValueError(
            f"최상위에서 허용되지 않은 필드가 발견되었습니다: {extra_keys}. " f"허용 필드: {allowed_keys}"
        )

    # 5) 타입 검사
    #   references는 list, nodes는 list, edges는 list인지(프로젝트 설계에 따라 달라질 수 있음)
    #   여기서는 예시로 references/nodes/edges를 list로 가정
    if not isinstance(data.get("references", []), list):
        raise ValueError("'references' 필드는 list 형식이어야 합니다.")
    if not isinstance(data.get("nodes", []), list):
        raise ValueError("'nodes' 필드는 list 형식이어야 합니다.")
    if not isinstance(data.get("edges", []), list):
        raise ValueError("'edges' 필드는 list 형식이어야 합니다.")

    # 여기까지 통과하면 최상위 구조는 references, nodes, edges만 있고, 타입도 맞음.


def validate_references(refs: List[Any]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    references 섹션(비실행 노드 목록)에 대해 유효성 검사:
      - 각 항목이 dict인지
      - 'name', 'type' 필드가 존재하는지
      - 중복 이름 여부
      - type이 NON_EXECUTION_TYPES에 속하는지
    반환: (non_exec_nodes, ref_names)

    * 비실행 노드는 BFS 검사 대상이 아니므로, 단순한 구조만 확인하면 됨.
    """
    ref_names: Set[str] = set()
    non_exec_nodes: List[Dict[str, Any]] = []

    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise ValueError(f"references[{i}]가 dict 형태가 아닙니다.")

        name = ref.get("name")
        rtype = ref.get("type")
        if not name or not isinstance(name, str):
            raise ValueError(f"references[{i}]에 'name'이 없거나 문자열이 아닙니다.")
        if name in ref_names:
            raise ValueError(f"references에서 노드 이름이 중복되었습니다: '{name}'")
        ref_names.add(name)

        if not rtype or not isinstance(rtype, str):
            raise ValueError(f"references[{i}] ('{name}')에 'type'이 없거나 문자열이 아닙니다.")
        if rtype not in NON_EXECUTION_TYPES:
            raise ValueError(
                f"'{name}': 비실행 노드 타입이 '{rtype}'로 지정됐으나, "
                f"NON_EXECUTION_TYPES에 포함되지 않습니다: {NON_EXECUTION_TYPES}"
            )

        non_exec_nodes.append(ref)

    return non_exec_nodes, ref_names


def validate_nodes(
    nodes: List[Any],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    nodes 섹션(주로 실행 노드)에 대해:
      - 각 항목이 dict인지
      - 'name', 'type' 필드가 유효한지
      - 노드 이름 중복 여부
      - type이 EXECUTION_TYPES에 속해야 함 (비실행 노드는 허용 안 함)
    반환값: (execution_nodes, node_names)

    * 만약 비실행 노드 타입(NON_EXECUTION_TYPES)이 들어오면 에러를 발생시킴.
    """
    node_names: Set[str] = set()
    execution_nodes: List[Dict[str, Any]] = []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{i}]가 dict 형태가 아닙니다.")

        name = node.get("name")
        ntype = node.get("type")

        if not name or not isinstance(name, str):
            raise ValueError(f"nodes[{i}]에 'name'이 없거나 문자열이 아닙니다.")
        if name in node_names:
            raise ValueError(f"노드 이름이 중복되었습니다: '{name}'")
        node_names.add(name)

        if not ntype or not isinstance(ntype, str):
            raise ValueError(f"{name}: 'type'이 없거나 문자열이 아닙니다.")

        # from_yaml 특수 처리
        if ntype == "from_yaml":
            validate_from_yaml_node(node)

        # 실행 노드 타입만 허용
        if ntype in EXECUTION_TYPES:
            execution_nodes.append(node)
        elif ntype in NON_EXECUTION_TYPES:
            raise ValueError(
                f"'{name}' 노드가 '{ntype}'로 지정되어 있으나, 이는 비실행 노드 타입이므로 "
                "nodes 섹션에서는 허용되지 않습니다. references 섹션에 선언해야 합니다."
            )
        else:
            raise ValueError(
                f"지원하지 않는 노드 유형: '{ntype}'. "
                f"허용되는 type: {EXECUTION_TYPES | NON_EXECUTION_TYPES}"
            )

    return execution_nodes, node_names


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
    adjacency["START"] = []

    for edge in edges:
        fr = edge["from"]
        to = edge["to"]
        if fr == "START":
            if to in execution_node_names:
                adjacency["START"].append(to)
        elif to == "END":
            if fr in execution_node_names:
                adjacency[fr].append("END")
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
    1) 노드가 여러 개 있는지 (최소 1개 이상)
    2) END로 향하는 edge가 정확히 하나만 존재하는지 확인
       - edges 중 to=END인 edge가 여러 개이면 에러
       - 없으면 에러
    """
    if len(nodes) < 1:
        raise ValueError("노드가 최소 1개 이상이어야 합니다.")

    end_count = sum(1 for e in edges if e.get("to") == "END")
    # 실행 노드가 있을 경우, END edge가 반드시 1개여야 함
    if end_count == 0 and len(edges) > 0:
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


def validate_config_param_key(data: dict, unify_key: str = "param"):
    """
    references와 nodes 섹션을 순회하며,
    각 항목의 config에 'params'가 발견되면 에러.
    'param'만 허용.
    만약 'param'마저 누락 시 에러로 볼지 여부는 하단 주석 참고.

    unify_key: 최종 허용 키 (기본 "param")
    """

    references = data.get("references", [])
    for ref_def in references:
        cfg = ref_def.get("config", {})
        check_param_key(cfg, unify_key, is_reference=True, name=ref_def.get("name"))

    nodes = data.get("nodes", [])
    for node_def in nodes:
        cfg = node_def.get("config", {})
        check_param_key(cfg, unify_key, is_reference=False, name=node_def.get("name"))


def check_param_key(cfg: dict, unify_key: str, is_reference: bool, name: str):
    """
    cfg: 해당 레퍼런스/노드의 config dict
    unify_key: "param" 등 허용된 키
    is_reference: True면 references 섹션, False면 nodes 섹션
    name: 레퍼런스나 노드 이름 (에러 메시지용)
    """
    has_params = "params" in cfg

    # 'params'가 있으면 에러
    if has_params:
        raise ValueError(
            f"{'Reference' if is_reference else 'Node'} '{name}'의 config에 "
            f"'params' 키가 발견되었습니다. '{unify_key}'로 통일해야 합니다."
        )


def validate_yaml(yaml_path: str) -> None:
    """
    references + nodes + edges 구조를 반영한 validate 함수.

    1) YAML 로드
    2) references(비실행 노드) 유효성 검사
    3) nodes(실행 노드) 유효성 검사 (비실행 노드는 허용하지 않음)
    4) edges 검사
    5) BFS 검사(실행 노드만)
    """
    data = load_yaml(yaml_path)
    # top level에 references, nodes, edges 외의 구조가 존재하는지 검증
    validate_top_level_structure(data)

    # references가 없으면 빈 리스트
    references = data.get("references", [])
    if not isinstance(references, list):
        raise ValueError("'references' 필드가 list 형태가 아닙니다.")

    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("'nodes' 필드가 list 형태가 아닙니다.")

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("'edges' 필드가 list 형태가 아닙니다.")

    # 1) references 검증
    non_exec_refs, ref_names = validate_references(references)

    # 2) nodes 검증 (비실행 노드 불허)
    execution_nodes, node_names = validate_nodes(nodes)

    # references 섹션에 있는 이름과 nodes 섹션에 있는 이름이 겹치면 에러
    dup = ref_names & node_names
    if dup:
        raise ValueError(f"references와 nodes에서 이름이 중복되었습니다: {dup}")

    # 3) edges 검사
    if len(execution_nodes) == 0:
        # 실행 노드가 없다면(=비실행 노드만 있다면) → edges가 없어도 괜찮다.
        if len(edges) > 0:
            # 혹시 edges가 있으면 엄격 모드로는 에러 처리 가능. 여기선 warning 취급 가능.
            pass
        return

    # 4) 내부 파라미터 키 검사
    validate_config_param_key(data)

    has_start_edge, has_end_edge = validate_edges(edges, node_names)
    if not has_start_edge:
        raise ValueError("그래프에 START에서 시작하는 edge가 없습니다.")
    if not has_end_edge:
        raise ValueError("그래프에 END로 가는 edge가 없습니다.")

    validate_bfs_for_execution_nodes(execution_nodes, edges, node_names)
    check_multiple_nodes_and_single_end(nodes, edges)
    check_all_nodes_reach_end(execution_nodes, edges, node_names)

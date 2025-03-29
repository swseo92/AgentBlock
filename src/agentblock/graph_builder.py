import os
import yaml
from typing import TypedDict, Type, Set, Any, Dict

from langgraph.graph import StateGraph, START, END

from agentblock.llm.llm_node import LLMNode
from agentblock.function.function_from_file_node import FunctionFromFileNode
from agentblock.function.function_from_library_node import FunctionFromLibraryNode
from agentblock.retriever.retriever_node import RetrieverNode
from agentblock.data_loader.base import GenericLoaderNode
from agentblock.vector_store.data_saver_node import DataSaverNode

from agentblock.embedding.embedding_reference import EmbeddingReference
from agentblock.vector_store.vector_store_reference import VectorStoreReference
from agentblock.schema.tools import validate_yaml

# 실행 노드 타입 매핑
NODE_TYPE_MAP = {
    "llm": LLMNode,
    "function_from_file": FunctionFromFileNode,
    "function_from_library": FunctionFromLibraryNode,
    "from_yaml": "handled separately",
    "retriever": RetrieverNode,
    "data_loader": GenericLoaderNode,
    "data_saver": DataSaverNode
    # 필요하면 "router" 등 다른 실행 노드 추가
}


class GraphBuilder:
    def __init__(self, path: str):
        """
        path가 YAML 파일 경로라고 가정.
        """
        self.yaml_path = os.path.abspath(path)
        self.validate_yaml()

        self.yaml_dir = os.path.dirname(self.yaml_path)
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 스키마에서 sections 파싱
        self.references_defs = self.config.get("references", [])
        self.node_defs = self.config.get("nodes", [])
        self.edge_defs = self.config.get("edges", [])

        # 저장 구조
        self.node_map = {}  # { node_name: node_fn or sub_graph }
        self.references_map: Dict[str, Any] = {}  # { reference_name: built_object }
        self.used_keys: Set[Any] = set()

    def validate_yaml(self):
        validate_yaml(self.yaml_path)

    def load_nodes(self):
        """
        nodes 섹션(실행 노드) 파싱 -> 실제 node_fn이나 subgraph 생성.
        - 각 노드는 from_yaml(..., references_map=self.references_map) 가능
        """
        for node_cfg in self.node_defs:
            node_type = node_cfg["type"]

            if node_type == "from_yaml":
                # 하위 YAML 로딩(재귀)
                from_file = node_cfg["config"].get("from_file")
                if not from_file:
                    raise ValueError(
                        f"{node_cfg['name']}: from_yaml node but no from_file specified"
                    )
                sub_file_path = os.path.join(self.yaml_dir, from_file)
                sub_builder = GraphBuilder(sub_file_path)
                # 재귀 빌드
                sub_graph = sub_builder.build_graph()
                self.node_map[node_cfg["name"]] = sub_graph

                # 서브그래프의 used_keys를 상위 그래프에도 반영
                self.used_keys.update(sub_builder.used_keys)

            else:
                # 일반 실행 노드
                cls = NODE_TYPE_MAP.get(node_type)
                if not cls or cls == "handled separately":
                    raise ValueError(f"Unsupported or special node type: {node_type}")

                # Node 클래스가 from_yaml(config, base_dir, references_map) 시그니처를 지원한다고 가정
                node_obj = cls.from_yaml(
                    node_cfg, base_dir=self.yaml_dir, references_map=self.references_map
                )
                node_fn = node_obj.build()
                self.node_map[node_cfg["name"]] = node_fn

                # input_keys / output_key -> used_keys 추가
                self.used_keys.update(node_cfg.get("input_keys", []))
                out_key = node_cfg.get("output_key")
                if isinstance(out_key, str):
                    self.used_keys.add(out_key)
                elif isinstance(out_key, list):
                    for k in out_key:
                        self.used_keys.add(k)

    def build_graph(self):
        """
        1) references 빌드 -> self.references_map
        2) nodes 빌드 -> self.node_map
        3) edges -> StateGraph
        """
        # 1) references 빌드
        self.load_references_topo()

        # 2) 실행 노드 빌드
        self.load_nodes()

        # 3) State TypedDict
        State = self.generate_state()

        # 4) StateGraph 구성
        graph = StateGraph(State)

        # 5) add_node
        for name, fn in self.node_map.items():
            graph.add_node(name, fn)

        # 6) edges
        for edge in self.edge_defs:
            from_name = edge["from"]
            to_name = edge["to"]
            if from_name == "START":
                from_name = START
            if to_name == "END":
                to_name = END

            if "condition" in edge:
                graph.add_conditional_edges(
                    from_name,
                    lambda state: state.get("route"),
                    {edge["condition"]: to_name},
                )
            else:
                graph.add_edge(from_name, to_name)

        return graph.compile()

    def generate_state(self) -> Type[TypedDict]:
        """
        used_keys를 기반으로 TypedDict 타입을 동적으로 생성
        """
        state_dict = {}
        for k in self.used_keys:
            state_dict[k] = Any
        return TypedDict("State", state_dict, total=False)

    def load_references_topo(self):
        """
        references 섹션의 의존 관계를 바탕으로 DAG를 만들고,
        Topological Sort로 순서대로 build한다.

        예:
          my_embedding  -> depends on no one
          my_vector_store -> depends on my_embedding
        """
        # 1) Parse references into a dict: name -> ref_def
        name_to_refdef = {}
        for ref_def in self.references_defs:
            ref_name = ref_def["name"]
            name_to_refdef[ref_name] = ref_def

        # 2) Build adjacency (dependency graph)
        #    edges: if refB depends on refA -> A->B
        graph = {}  # adjacency dict: refName -> list of references that depend on it
        in_degree = {}  # each reference's in-degree count

        for ref_name, ref_def in name_to_refdef.items():
            graph[ref_name] = []
            in_degree[ref_name] = 0

        # find dependencies
        for ref_name, ref_def in name_to_refdef.items():
            cfg = ref_def.get("config", {})
            sub_refs = cfg.get("reference", {})  # e.g. {"embedding": "my_embedding"}
            # sub_refs.values() = ["my_embedding", ...]
            for dep_name in sub_refs.values():
                # if dep_name is a reference name, then dep_name -> ref_name
                if dep_name in name_to_refdef:
                    # add edge dep_name -> ref_name
                    graph[dep_name].append(ref_name)
                    in_degree[ref_name] += 1

        # 3) Kahn's Algorithm or similar topological sort
        from collections import deque

        queue = deque()

        # queue에 in_degree=0인 노드(의존 없는 레퍼런스) 넣기
        for ref_name, deg in in_degree.items():
            if deg == 0:
                queue.append(ref_name)

        topo_order = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)

            # 그래프에서 current -> next
            for nxt in graph[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # 에러 체크: 만약 topo_order 길이가 전체 refs 개수보다 작으면, 순환 의존이 존재
        if len(topo_order) < len(name_to_refdef):
            raise ValueError(
                "Reference cyclic dependency detected. Could not topologically sort."
            )

        # 4) 실제 build 순서대로 진행
        for ref_name in topo_order:
            ref_def = name_to_refdef[ref_name]
            ref_type = ref_def["type"]

            if ref_type == "embedding":
                emb_ref = EmbeddingReference.from_yaml(
                    ref_def, base_dir=self.yaml_dir, references_map=self.references_map
                )
                built_obj = emb_ref.build()
                self.references_map[ref_name] = built_obj

            elif ref_type == "vector_store":
                vs_ref = VectorStoreReference.from_yaml(
                    ref_def, base_dir=self.yaml_dir, references_map=self.references_map
                )
                built_obj = vs_ref.build()
                self.references_map[ref_name] = built_obj

            else:
                # other references or skip
                pass

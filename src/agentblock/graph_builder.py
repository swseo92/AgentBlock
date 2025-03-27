import yaml
from typing import TypedDict, Type, Set
from langgraph.graph import StateGraph

# START, END

# 노드 구현체
from agentblock.llm.llm_node import LLMNode

# from agentblock.function_node import FunctionNode

NODE_TYPE_MAP = {
    "llm": LLMNode,
    # "function": FunctionNode,
    # "retriever": RetrieverNode, ...
    "from_yaml": "handled separately",
}


class GraphBuilder:
    def __init__(self, path_or_dict: str | dict):
        if isinstance(path_or_dict, str):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = path_or_dict

        self.node_defs = self.config["nodes"]
        self.edge_defs = self.config.get("edges", [])
        self.node_map = {}  # name → node_fn
        self.used_keys: Set[str] = set()

    def generate_state(self) -> Type[TypedDict]:
        state_dict = {k: (str, ...) for k in self.used_keys}
        return TypedDict("State", state_dict, total=False)

    def load_nodes(self):
        for config in self.node_defs:
            node_type = config["type"]

            if node_type == "from_yaml":
                # 재귀적으로 그래프 로딩
                with open(config["from_file"], "r", encoding="utf-8") as f:
                    sub_config = yaml.safe_load(f)

                # 하위 그래프 빌드
                sub_builder = GraphBuilder(sub_config)
                sub_graph = sub_builder.build_graph()

                self.node_map[config["name"]] = sub_graph
                self.used_keys.update(sub_builder.used_keys)

            else:
                cls = NODE_TYPE_MAP.get(node_type)
                if not cls:
                    raise ValueError(f"Unsupported node type: {node_type}")

                node = cls.from_yaml(config)
                self.node_map[config["name"]] = node.build()

                self.used_keys.update(config.get("input_keys", []))
                if config.get("output_key"):
                    self.used_keys.add(config["output_key"])

    def build_graph(self):
        self.load_nodes()
        State = self.generate_state()
        graph = StateGraph(State)

        for name, fn in self.node_map.items():
            graph.add_node(name, fn)

        for edge in self.edge_defs:
            if "condition" in edge:
                graph.add_conditional_edges(
                    edge["from"],
                    lambda state: state.get("route"),  # 개선 가능
                    {edge["condition"]: edge["to"]},
                )
            else:
                graph.add_edge(edge["from"], edge["to"])

        return graph.compile()

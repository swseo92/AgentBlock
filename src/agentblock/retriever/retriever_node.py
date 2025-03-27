from agentblock.base import BaseNode


class RetrieverNode(BaseNode):
    def __init__(
        self, name, retriever, input_key="query", output_key="docs", kwargs=None
    ):
        super().__init__(name)
        self.retriever = retriever
        self.input_key = input_key
        self.output_key = output_key
        self.kwargs = kwargs or {}

    @staticmethod
    def from_yaml(config: dict, retriever_map: dict) -> "RetrieverNode":
        return RetrieverNode(
            name=config["name"],
            retriever=retriever_map[config["retriever_name"]],
            input_key=config.get("input_key", "query"),
            output_key=config.get("output_key", "docs"),
            kwargs=config.get("kwargs", {}),
        )

    def build(self):
        def node_fn(state):
            query = state[self.input_key]
            docs = self.retriever.get_relevant_documents(query, **self.kwargs)
            return {self.output_key: docs}

        return node_fn

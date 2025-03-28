from agentblock.function import FunctionFromFileNode


class DataLoaderFromFileNode(FunctionFromFileNode):
    """
    별도 .py 파일의 로딩 함수를 function_path로 지정해,
    문서를 읽어 list[Document] 형태로 반환하는 Node.
    """

    @staticmethod
    def from_yaml(config: dict, base_dir: str = None) -> "DataLoaderFromFileNode":
        return DataLoaderFromFileNode(
            name=config["name"],
            function_path=config["config"]["function_path"],
            input_keys=config.get("input_keys", []),
            output_key=config["output_key"],
            base_dir=base_dir,
            params=config["config"].get("params", {}),
        )

    def build(self):
        """
        부모 FunctionNode의 build() 로직을 그대로 활용.
        (function_path에서 함수를 로딩해, dict 반환을 기대)
        """
        return super().build()

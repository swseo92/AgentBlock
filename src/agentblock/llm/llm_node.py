from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from agentblock.base import BaseNode
from agentblock.llm.llm_factory import LLMFactory
from typing import Dict


class LLMNode(BaseNode):
    def __init__(
        self,
        name=None,
        provider=None,
        args=None,
        kwargs=None,
        prompt_template=None,
        input_keys=None,
        output_key=None,
    ):
        super().__init__(name)
        self.provider = provider
        self.prompt_template = prompt_template
        self.input_keys = input_keys
        self.output_key = output_key
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def from_yaml(config: dict, base_dir=None) -> "LLMNode":
        config_llm = config["config"]
        return LLMNode(
            name=config["name"],
            input_keys=config["input_keys"],
            output_key=config["output_key"],
            provider=config_llm["provider"],
            args=config_llm["args"],
            kwargs=config_llm["kwargs"],
            prompt_template=config_llm["prompt_template"],
        )

    def build(self):
        prompt = PromptTemplate.from_template(self.prompt_template)
        if self.args is None:
            llm = LLMFactory().create_llm(provider=self.provider, **self.kwargs)
        else:
            llm = LLMFactory().create_llm(
                provider=self.provider, *self.args, **self.kwargs
            )

        chain = LLMChain(prompt=prompt, llm=llm, output_key=self.output_key)

        def node_fn(state: Dict) -> Dict:
            # 입력값 준비
            inputs = self.get_inputs(state)
            result = chain(inputs)
            return {self.output_key: result[self.output_key]}

        return node_fn


if __name__ == "__main__":
    from dotenv import load_dotenv
    from agentblock.schema.tools import validate_yaml

    load_dotenv()

    path_yaml = "template_llm.yaml"
    validate_yaml(path_yaml)

    node = LLMNode().from_yaml_file_single_node(path_yaml).build()
    result = node({"query": "hi"})
    print(result)

    assert "law_response" in result.keys(), result
    assert result["law_response"], result

    from agentblock.graph_builder import GraphBuilder

    graph = GraphBuilder(path_yaml).build_graph()

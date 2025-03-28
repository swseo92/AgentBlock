from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from agentblock.base import BaseNode
from agentblock.llm.llm_factory import LLMFactory
from typing import Dict, Any


class LLMNode(BaseNode):
    def __init__(
        self,
        name=None,
        provider=None,
        param=None,
        prompt_template=None,
        input_keys=None,
        output_key=None,
    ):
        super().__init__(name)
        self.provider = provider
        self.prompt_template = prompt_template
        self.input_keys = input_keys
        self.output_key = output_key
        self.param = param

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str = None, references_map: Dict[str, Any] = None
    ) -> "LLMNode":
        return LLMNode(
            name=config["name"],
            input_keys=config["input_keys"],
            output_key=config["output_key"],
            provider=config["config"]["provider"],
            param=config["config"]["param"],
            prompt_template=config["config"]["prompt_template"],
        )

    def build(self):
        prompt = PromptTemplate.from_template(self.prompt_template)
        llm = LLMFactory().create_llm(provider=self.provider, **self.param)

        chain = LLMChain(prompt=prompt, llm=llm, output_key=self.output_key)

        def node_fn(state: Dict) -> Dict:
            # 입력값 준비
            inputs = self.get_inputs(state)
            result = chain(inputs)
            return {self.output_key: result[self.output_key]}

        return node_fn

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from agentblock.base import BaseNode
from agentblock.llm.llm_factory import LLMFactory


class LLMNode(BaseNode):
    def __init__(
        self, name, provider, args, kwargs, prompt_template, input_keys, output_key
    ):
        super().__init__(name)
        self.provider = provider
        self.prompt_template = prompt_template
        self.input_keys = input_keys
        self.output_key = output_key
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def from_yaml(config: dict) -> "LLMNode":
        return LLMNode(
            name=config["name"],
            provider=config["provider"],
            args=config["args"],
            kwargs=config["kwargs"],
            prompt_template=config["prompt_template"],
            input_keys=config["input_keys"],
            output_key=config["output_key"],
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

        def node_fn(state):
            inputs = {k: state[k] for k in self.input_keys}
            result = chain(inputs)
            return {self.output_key: result[self.output_key]}

        return node_fn

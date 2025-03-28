import pytest
from agentblock.llm.llm_factory import LLMFactory
from langchain_openai import ChatOpenAI
from agentblock.tools.load_config import load_config
from agentblock.sample_data.tools import get_sample_data

from dotenv import load_dotenv


path_config = get_sample_data("yaml/llm/legal_assistant.yaml")


def test_llm_factory_langchain():
    load_dotenv()
    config = load_config(path_config)
    config_node = config["nodes"][0]
    provider = config_node["config"]["provider"]
    param = config_node["config"]["param"]

    llm = LLMFactory.create_llm(provider=provider, **param)
    assert isinstance(llm, ChatOpenAI)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm(
            provider="invalid", model="gpt-4o-mini", temperature=0.0
        )

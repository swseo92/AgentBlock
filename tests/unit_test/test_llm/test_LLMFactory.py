import pytest
from agentblock.llm.LLMFactory import LLMFactory
from langchain_openai import ChatOpenAI
from agentblock.tools.load_config import load_config

from dotenv import load_dotenv

path_config = "../test_config.yaml"


def test_llm_factory_langchain():
    load_dotenv()
    config = load_config(path_config)

    llm = LLMFactory.create_llm(**config["llm"])
    assert isinstance(llm, ChatOpenAI)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm(
            provider="invalid", model="gpt-3.5-turbo", temperature=0.0
        )

import pytest
from agentblock.llm.llm_factory import LLMFactory
from langchain_openai import ChatOpenAI
from agentblock.tools.load_config import load_config

from dotenv import load_dotenv
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


path_config = "../test_config.yaml"


def test_llm_factory_langchain():
    load_dotenv()
    config = load_config(path_config)

    llm = LLMFactory.create_llm(**config["llm"])
    assert isinstance(llm, ChatOpenAI)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm(
            provider="invalid", model="gpt-4o-mini", temperature=0.0
        )

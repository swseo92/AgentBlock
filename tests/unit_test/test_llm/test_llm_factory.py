import pytest
from agentblock.llm.llm_factory import LLMFactory
from langchain_openai import ChatOpenAI
from agentblock.tools.load_config import load_config

from dotenv import load_dotenv
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


path_config = "./test_llm.yaml"


def test_llm_factory_langchain():
    load_dotenv()
    config = load_config(path_config)
    config_node = config["nodes"][0]
    provider = config_node["config"]["provider"]
    args = config_node["config"]["args"]
    kwargs = config_node["config"]["kwargs"]

    llm = LLMFactory.create_llm(*args, provider=provider, **kwargs)
    assert isinstance(llm, ChatOpenAI)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm(
            provider="invalid", model="gpt-4o-mini", temperature=0.0
        )

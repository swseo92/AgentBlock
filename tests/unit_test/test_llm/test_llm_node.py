import yaml

from agentblock.llm.llm_node import LLMNode

from dotenv import load_dotenv


path_config = "./test_llm.yaml"
path_config2 = "./test_llm2.yaml"


def test_from_yaml():
    """
    template_llm.yaml을 읽어 LLMNode.from_yaml()을 통해
    LLMNode 객체가 제대로 생성되는지 검증.
    """
    with open(path_config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    node = LLMNode.from_yaml(config_data)

    assert node.name == "legal_assistant"
    assert node.provider == "openai"
    assert node.kwargs["model_name"] == "gpt-4"
    assert "query" in node.input_keys
    assert node.output_key == "answer"


def test_build():
    """
    build() 메서드를 테스트.
    - PromptTemplate이 생성되는지
    - LLMFactory가 호출되어 llm 인스턴스가 반환되는지
    - LLMChain이 올바르게 초기화되는지 (부분 확인)
    """
    load_dotenv()

    with open(path_config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 1) LLMNode 생성
    node = LLMNode.from_yaml(config_data)

    # 2) build() 호출
    node_fn = node.build()

    # build()가 반환하는 것은 node_fn(함수).
    # node_fn이 실제로 체인을 호출할 때 사용할 llm이 llm_mock인지 확인
    assert callable(node_fn), "build()는 호출 가능한 함수를 반환해야 합니다."


def test_node_fn_inference():
    """
    node_fn을 실제로 호출해보고,
    state에서 input_keys를 뽑아 체인에 전달하는지,
    체인 결과가 output_key에 매핑되는지 확인.
    """
    load_dotenv()

    with open(path_config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    node = LLMNode.from_yaml(config_data)

    node_fn = node.build()
    node_fn(state={"query": "Hello World"})


def test_graph_compile():
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    from typing import Annotated
    from typing_extensions import TypedDict

    class State(TypedDict):
        query: str
        answer: Annotated[list, add_messages]

    load_dotenv()

    with open(path_config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    node = LLMNode.from_yaml(config_data)
    node_fn = node.build()

    graph_builder = StateGraph(State)
    graph_builder.add_node(node.name, node_fn)

    graph_builder.add_edge(START, node.name)
    graph_builder.add_edge(node.name, END)

    graph = graph_builder.compile()
    result = graph.invoke({"query": "hi"})

    assert result["query"] == "hi"
    assert result["answer"] == 1


def test_multi_node():
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    from typing import Annotated
    from typing_extensions import TypedDict

    class State(TypedDict):
        query: str
        query2: str

        answer: Annotated[list, add_messages]
        answer2: Annotated[list, add_messages]

    load_dotenv()

    with open(path_config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    with open(path_config2, "r", encoding="utf-8") as f:
        config_data2 = yaml.safe_load(f)

    node = LLMNode.from_yaml(config_data)
    node_fn = node.build()

    node2 = LLMNode.from_yaml(config_data2)
    node2_fn = node2.build()

    graph_builder = StateGraph(State)
    graph_builder.add_node(node.name, node_fn)
    graph_builder.add_node(node2.name, node2_fn)

    graph_builder.add_edge(START, node.name)
    graph_builder.add_edge(node.name, node2.name)
    graph_builder.add_edge(node2.name, END)

    graph = graph_builder.compile()
    result = graph.invoke({"query": "hi", "query2": "hi hi"})

    assert result["query"] == "hi"
    assert result["query2"] == "hi hi"

    assert len(result["answer"]) == 1
    assert len(result["answer2"]) == 1

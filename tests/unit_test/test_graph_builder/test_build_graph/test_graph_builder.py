from agentblock.graph_builder import GraphBuilder
from dotenv import load_dotenv
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

load_dotenv()

base_path = script_dir
path_main_graph = f"{base_path}/main_graph.yaml"


def test_graph_from_recursive_yaml():
    # 1. 메인 그래프 YAML 경로
    path = path_main_graph

    # 2. GraphBuilder 생성 및 컴파일
    builder = GraphBuilder(path)
    graph = builder.build_graph()

    # 3. 입력 데이터 (State)
    test_input = {"query": "안녕하세요"}

    # 4. 실행
    result = graph.invoke(test_input)

    # 5. 결과 검사
    assert "final_answer" in result
    assert isinstance(result["final_answer"], str)
    assert len(result["final_answer"]) > 0

    # 6. State 체크 (자동 생성 키 포함 여부)
    assert "query" in result


def test_graph_builds_successfully():
    builder = GraphBuilder(path_main_graph)
    graph = builder.build_graph()
    assert graph is not None


def test_graph_runs_with_valid_input():
    builder = GraphBuilder(path_main_graph)
    graph = builder.build_graph()

    result = graph.invoke({"query": "안녕하세요"})
    assert "final_answer" in result
    assert isinstance(result["final_answer"], str)
    assert len(result["final_answer"]) > 0


def test_recursive_state_keys():
    builder = GraphBuilder(path_main_graph)
    builder.build_graph()

    expected_keys = {"query", "law_response", "law_summary", "final_answer"}
    assert expected_keys.issubset(builder.used_keys)


def test_output_key_integrity():
    builder = GraphBuilder(path_main_graph)
    graph = builder.build_graph()

    result = graph.invoke({"query": "안녕하세요"})
    assert "law_response" in result
    assert "law_summary" in result
    assert "final_answer" in result

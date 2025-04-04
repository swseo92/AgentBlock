from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data

from dotenv import load_dotenv

load_dotenv()


path_main_graph = get_sample_data("graph/graph_for_test/main_graph.yaml")

# 아래와 같은 구조를 테스트함
# 테스트 범위
# - GraphBuilder를 통한 파이프라인 조립
# - llm, function 객체 검증
# - 서브그래프 구조

# (START)
#    ↓
# law_pipeline  (type: from_yaml, from_file=graphs/law_graph.yaml)
#    ↓
# llm_legal     (type: llm)
#    ↓
# summarizer    (type: llm)
#    ↓
# merger        (type: function)
#    ↓
# (END)


def test_graph_from_recursive_yaml():
    # 1. 메인 그래프 YAML 경로
    path = path_main_graph

    # 2. GraphBuilder 생성 및 컴파일
    builder = GraphBuilder(path)
    graph = builder.build()

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
    graph = builder.build()
    assert graph is not None


def test_graph_runs_with_valid_input():
    builder = GraphBuilder(path_main_graph)
    graph = builder.build()

    result = graph.invoke({"query": "안녕하세요"})
    assert "final_answer" in result
    assert isinstance(result["final_answer"], str)
    assert len(result["final_answer"]) > 0


def test_recursive_state_keys():
    builder = GraphBuilder(path_main_graph)
    builder.build()

    expected_keys = {"query", "law_response", "law_summary", "final_answer"}
    assert expected_keys.issubset(builder.used_keys)


def test_output_key_integrity():
    builder = GraphBuilder(path_main_graph)
    graph = builder.build()

    result = graph.invoke({"query": "안녕하세요"})
    assert "law_response" in result
    assert "law_summary" in result
    assert "final_answer" in result

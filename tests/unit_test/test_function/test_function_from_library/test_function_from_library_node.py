import pytest
from agentblock.graph_builder import GraphBuilder


def test_function_from_library_yaml(tmp_path):
    """
    테스트 시나리오:
    1) 임시 디렉토리에 some_library.py, function_from_library_test.yaml 파일을 생성
    2) GraphBuilder로 YAML 빌드 → 그래프 실행
    3) state={"a":2, "b":3, "x":5, "y":10} 넣으면 20을 기대
    """

    # 1) 임시 파일 생성: some_library.py
    library_code = """
def sum_cal(a, b, x=0, y=0):
    return a + b + x + y
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    # 2) 임시 파일 생성: function_from_library_test.yaml
    yaml_text = """\
nodes:
  - name: sum_cal_node
    type: function_from_library
    input_keys:
      - a
      - b
      - x
      - y
    output_key: result
    config:
      from_library: "./some_library:sum_cal"
      params: []

edges:
  - from: START
    to: sum_cal_node
  - from: sum_cal_node
    to: END
"""
    yaml_file = tmp_path / "function_from_library_test.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    # 3) GraphBuilder로 빌드
    builder = GraphBuilder(str(yaml_file))
    graph = builder.build_graph()

    # 4) 그래프 실행
    state = {"a": 2, "b": 3, "x": 5, "y": 10}
    output_state = graph.invoke(state)

    # 5) 결과 검증
    assert "result" in output_state, "Output must contain 'result'"
    assert output_state["result"] == 20, f"Expected 20, got {output_state['result']}"

    print("Test passed: function_from_library_yaml => result=20 as expected")


def test_missing_file(tmp_path):
    """
    잘못된 from_library 경로 -> FileNotFoundError 발생 예상
    """
    # 빈 library 파일 X
    yaml_text = """\
nodes:
  - name: sum_cal_node
    type: function_from_library
    input_keys: [a, b]
    output_key: result
    config:
      from_library: "./not_exist:sum_cal"
      params: []
edges:
  - from: START
    to: sum_cal_node
  - from: sum_cal_node
    to: END
"""
    yaml_file = tmp_path / "bad_from_library.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    builder = GraphBuilder(str(yaml_file))

    with pytest.raises(FileNotFoundError):
        _ = builder.build_graph()


def test_multiple_output_with_params(tmp_path):
    """
    (a, b, x, y)로 계산하여 (합, 차이, 곱, 나눗셈) 등 다양한 결과를 tuple로 반환.
    output_key가 여러 개인 경우.
    """
    library_code = r"""
def multi_ops(a, b, x=0, y=0):
    # 예: (합, 차, 곱, 몫)
    s = a + b + x + y
    d = a - b
    mul = a*b + x*y
    # b==0 시 나눗셈 예외 가능 => 단순 처리
    div = a/b if b != 0 else None
    return (s, d, mul, div)
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: multi_ops_node
    type: function_from_library
    input_keys: [a, b, x, y]
    output_key: [sum, diff, product, quotient]
    config:
      from_library: "./some_library:multi_ops"
      params: []
edges:
  - from: START
    to: multi_ops_node
  - from: multi_ops_node
    to: END
"""
    yaml_file = tmp_path / "function_from_library_test.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    builder = GraphBuilder(str(yaml_file))
    graph = builder.build_graph()

    state = {"a": 4, "b": 2, "x": 3, "y": 1}
    # multi_ops(4,2,x=3,y=1):
    # sum=4+2+3+1=10, diff=4-2=2, product=4*2 + 3*1=8+3=11, quotient=4/2=2
    result_state = graph.invoke(state)

    assert result_state["sum"] == 10
    assert result_state["diff"] == 2
    assert result_state["product"] == 11
    assert result_state["quotient"] == 2


def test_multiple_output_length_mismatch(tmp_path):
    """
    라이브러리 함수가 2개 요소의 tuple을 반환하지만,
    output_key를 3개로 선언해 길이 불일치 -> ValueError 예상
    """
    library_code = r"""
def short_ops(a, b):
    return (a+b, a*b)
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: short_ops_node
    type: function_from_library
    input_keys: [a, b]
    output_key: [sum, product, difference]  # 3개 키
    config:
      from_library: "./some_library:short_ops"
      params: []
edges:
  - from: START
    to: short_ops_node
  - from: short_ops_node
    to: END
"""
    yaml_file = tmp_path / "function_from_library_test.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    builder = GraphBuilder(str(yaml_file))

    graph = builder.build_graph()
    state = {"a": 4, "b": 3}

    with pytest.raises(ValueError) as excinfo:
        graph.invoke(state)

    # 메시지 확인
    assert "len mismatch" in str(excinfo.value), str(excinfo.value)


def test_params_override_input_keys(tmp_path):
    """
    if the same variable name is in input_keys and also in params,
    we might want to see how the node handles it.

    여기서는 'x'를 input_keys에도 넣고, params에도 넣어서,
    실제 호출에 어떤 값이 넘어가는지 확인할 수 있음.
    """
    library_code = r"""
def debug_sum(a, x=0):
    return a + x
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: debug_sum_node
    type: function_from_library
    input_keys: [a, x]
    output_key: result
    config:
      from_library: "./some_library:debug_sum"
      params: ["x"]   # x를 다시 param에 명시
edges:
  - from: START
    to: debug_sum_node
  - from: debug_sum_node
    to: END
"""
    yaml_file = tmp_path / "debug_sum_test.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    builder = GraphBuilder(str(yaml_file))
    graph = builder.build_graph()

    # state 내에서 a=5, x=10으로 설정
    # input_keys에서 x=10을 가져오지만, params=["x"]를 또 가져오면?
    # test how the node merges them
    state = {"a": 5, "x": 10}
    result = graph.invoke(state)
    # debug_sum(a=5, x=10) => 15
    assert result["result"] == 15


def test_function_throws_exception(tmp_path):
    """
    라이브러리 함수 내부에서 ZeroDivisionError 등이 발생하는 시나리오.
    Node가 그대로 예외를 전파하면, graph.invoke()에서 해당 예외가 발생해야 함.
    """
    library_code = r"""
def risky_div(a, b):
    return a / b  # b=0 => ZeroDivisionError
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: div_node
    type: function_from_library
    input_keys: [a, b]
    output_key: result
    config:
      from_library: "./some_library:risky_div"
      params: []
edges:
  - from: START
    to: div_node
  - from: div_node
    to: END
"""
    yaml_file = tmp_path / "risky_div.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    builder = GraphBuilder(str(yaml_file))
    graph = builder.build_graph()

    # b=0 -> ZeroDivisionError 발생
    state = {"a": 5, "b": 0}

    with pytest.raises(ZeroDivisionError):
        graph.invoke(state)


def test_function_returns_none(tmp_path):
    """
    라이브러리 함수가 None을 반환할 경우,
    output_key: "result"에 {result: None}로 담길지 확인
    """
    library_code = r"""
def none_func(a, b):
    # 단순히 None 반환
    return None
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: none_node
    type: function_from_library
    input_keys: [a, b]
    output_key: result
    config:
      from_library: "./some_library:none_func"
      params: []
edges:
  - from: START
    to: none_node
  - from: none_node
    to: END
"""
    yaml_file = tmp_path / "none_func.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    graph = GraphBuilder(str(yaml_file)).build_graph()

    result_state = graph.invoke({"a": 1, "b": 2})
    # expect => {"result": None}
    assert "result" in result_state
    assert result_state["result"] is None


def test_function_params_not_in_state(tmp_path):
    """
    config.params에 명시된 'x','y'가 state에 없어도 오류 없이 동작해야 하는지,
    또는 key error를 일으켜야 하는지 확인.
    """
    library_code = """
def sum_cal(a, b, x=0, y=0):
    return a + b + x + y
"""
    lib_file = tmp_path / "some_library.py"
    lib_file.write_text(library_code, encoding="utf-8")

    yaml_text = """\
nodes:
  - name: sum_node
    type: function_from_library
    input_keys: [a, b]
    output_key: result
    config:
      from_library: "./some_library:sum_cal"
      params: ["x","y"]  # state에 x,y가 없을 때 => x=0,y=0?
edges:
  - from: START
    to: sum_node
  - from: sum_node
    to: END
"""
    yaml_file = tmp_path / "sum_cal.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    from agentblock.graph_builder import GraphBuilder

    graph = GraphBuilder(str(yaml_file)).build_graph()

    # state에 x,y 없음 => default=0
    result_state = graph.invoke({"a": 2, "b": 3})
    # => sum_cal(a=2,b=3,x=0,y=0)=5
    assert result_state["result"] == 5

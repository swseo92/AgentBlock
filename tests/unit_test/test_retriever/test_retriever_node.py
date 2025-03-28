import os
import shutil
import pytest
from agentblock.graph_builder import GraphBuilder
from agentblock.sample_data.tools import get_sample_data

_vector_store_map = {}  # name -> vector store object (if needed)


def set_vector_store(name: str, vs_obj):
    _vector_store_map[name] = vs_obj


def store_docs(docs_to_add, vector_store_name: str):
    """
    docs_to_add: list[str]
    vector_store_name: ex) "my_faiss"
    이 함수를 호출하면, vector_store에 문서를 추가하고 저장(disk)한다.
    """
    vs_obj = _vector_store_map.get(vector_store_name)
    if not vs_obj:
        raise ValueError(f"No vector store found for '{vector_store_name}'")

    # 문서 추가
    vs_obj.add_texts(docs_to_add)
    # 디스크 저장
    vs_obj.save()
    return {"status": "ok", "added_docs": len(docs_to_add)}


@pytest.fixture
def faiss_integration_yaml():
    path = get_sample_data("yaml/retriever/faiss_integration_test.yaml")
    return path


def remove_faiss_index(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def test_faiss_integration_flow(faiss_integration_yaml):
    from dotenv import load_dotenv

    load_dotenv()

    # 1) 인덱스 파일 제거 (test.faiss)
    index_path = "test.faiss"  # 실제 경로 맞춰 수정
    remove_faiss_index(index_path)

    # 2) GraphBuilder로 전체 YAML 빌드
    builder = GraphBuilder(faiss_integration_yaml)
    graph = builder.build_graph()

    # 3) references_map에 있는 vector_store 가져와 "store_docs" 함수가 접근할 수 있도록 set_vector_store
    #    - "my_faiss"는 references 섹션 name
    vs_obj = builder.references_map.get("my_faiss")
    assert vs_obj, "VectorStore 'my_faiss' should be built by references"
    set_vector_store("my_faiss", vs_obj)
    store_docs(
        docs_to_add=["Hello world", "Foo bar", "Embeddings are cool"],
        vector_store_name="my_faiss",
    )

    # 4) BFS 첫 실행: START -> storer_node -> my_retriever -> END
    #    storer_node input_keys=["docs_to_add"], my_retriever input_keys=["query"]
    #    => we pass docs_to_add + query
    input_state = {"query": "Hello"}
    final_state = graph.invoke(input_state)

    # 5) 결과 검증
    #    storer_node outputs => "store_result"
    #    retriever outputs => "retrieved_docs"
    assert "retrieved_docs" in final_state, "retriever output_key not in final state"
    docs = final_state["retrieved_docs"]
    print("[INFO] retrieved docs after BFS:", docs)
    assert isinstance(docs, list), "retrieved_docs must be a list"

    # 6) 인덱스가 disk에 저장됐는지 확인
    assert os.path.exists(index_path), "FAISS index file not found after BFS"

    builder2 = GraphBuilder(faiss_integration_yaml)
    graph2 = builder2.build_graph()
    final_state2 = graph2.invoke(input_state)

    assert final_state2 == final_state, "불러온 백터스토어는 동일한 결과를 출력해야합니다."

    # 8) 정리
    remove_faiss_index(index_path)

import copy
import os
import time
import shutil

from agentblock.tools.load_config import (
    get_yaml_for_single_node_file,
    get_parent_dir_abspath,
    get_abspath,
)
from agentblock.sample_data.tools import get_sample_data
from agentblock.retriever.retriever_node import (
    _load_embedding,
    _load_vector_store,
    RetrieverNode,
)


path_retriever_yaml = get_sample_data("yaml/retriever/test_retriever.yaml")


def get_vector_store(node_data, base_dir):
    emb_conf = node_data["config"]["embedding"]
    embedding_obj = _load_embedding(emb_conf, base_dir=base_dir)

    # 3) vector_store config 추출
    vs_conf = node_data["config"]["vector_store"]

    # 4) _load_vector_store 호출
    vs_obj = _load_vector_store(vs_conf, embedding_obj, base_dir=base_dir)
    return vs_obj


def get_vector_store_path_from_yaml(path_yaml):
    base_dir = get_parent_dir_abspath(path_yaml)
    node_data = get_yaml_for_single_node_file(path_yaml)

    path_faiss_yaml = node_data["config"]["vector_store"]["from_file"]
    path_abs_faiss_yaml = get_abspath(path_faiss_yaml, base_dir=base_dir)
    faiss_data = get_yaml_for_single_node_file(path_abs_faiss_yaml)

    base_dir_faiss = get_parent_dir_abspath(path_abs_faiss_yaml)
    path_faiss_index = faiss_data["config"]["path"]
    path_abs_faiss_index = get_abspath(path_faiss_index, base_dir=base_dir_faiss)
    return path_abs_faiss_index


def delete_faiss_index(path_faiss_index):
    if os.path.exists(path_faiss_index):
        shutil.rmtree(path_faiss_index)
        while True:
            time.sleep(0.1)
            if not os.path.exists(path_faiss_index):
                break


def test_load_embedding_from_file():
    """
    template_retriever.yaml에서 'my_retriever' 노드를 찾고,
    그 노드의 config.embedding을 이용해 _load_embedding을 호출.
    """
    node_data = get_yaml_for_single_node_file(path_retriever_yaml)

    # 1) embedding_conf 추출
    emb_conf = node_data["config"]["embedding"]

    # 2) _load_embedding 호출
    base_dir = get_parent_dir_abspath(path_retriever_yaml)
    embedding_obj = _load_embedding(emb_conf, base_dir=base_dir)

    # 3) 결과 검증
    # - embedding.yaml 내부가 어떻게 생겼는지에 따라 달라집니다.
    # - 예: "provider": "dummy", "dimension": 4 를 기대한다면:
    from agentblock.embedding.dummy_embedding import DummyEmbedding

    assert isinstance(
        embedding_obj, DummyEmbedding
    ), "from_file에 정의된 embedding은 DummyEmbedding이어야 함."
    # dimension도 확인 가능
    assert embedding_obj.dimension == 5


def test_load_vector_store_from_file():
    """
    template_retriever.yaml에서 'my_retriever' 노드를 찾고,
    그 노드의 config.vector_store를 이용해 _load_vector_store를 호출.
    """
    # 1) YAML 파일에서 단일 노드 정보를 읽어옴
    base_dir = get_parent_dir_abspath(path_retriever_yaml)
    node_data = get_yaml_for_single_node_file(path_retriever_yaml)

    vs_obj = get_vector_store(node_data, base_dir=base_dir)
    # 5) 결과 검증
    vs_obj.add_texts(["Hello world"])
    result = vs_obj.similarity_search("Hello")
    assert len(result) > 0

    # 6) 저장 후 로드 확인
    path_abs_faiss_index = get_vector_store_path_from_yaml(path_retriever_yaml)
    delete_faiss_index(path_abs_faiss_index)

    vs_obj.save()
    assert os.path.exists(path_abs_faiss_index)

    node_data2 = copy.deepcopy(node_data)
    vs_obj2 = get_vector_store(node_data2, base_dir=base_dir)

    result2 = vs_obj2.similarity_search("Hello")
    assert len(result2) > 0
    assert result2 == result

    delete_faiss_index(path_abs_faiss_index)


def test_retriever_node_from_yaml():
    """
    1) get_yaml_for_single_node_file(path_retriever_yaml)으로 'my_retriever' 노드 추출
    2) from_yaml(...) 호출 → RetrieverNode 인스턴스 생성
    3) 노드 필드(name, input_keys, output_key, config 등) 검사
    4) build() 및 node_fn까지 호출, 실제 검색 동작(빈 인덱스이면 빈 결과) 확인
    """
    # 1) YAML에서 단일 노드( my_retriever ) 추출
    node_data = get_yaml_for_single_node_file(path_retriever_yaml)
    base_dir = get_parent_dir_abspath(path_retriever_yaml)

    # 2) from_yaml으로 RetrieverNode 생성
    node = RetrieverNode.from_yaml(node_data, base_dir=base_dir)
    node_fn = node.build()

    # 3) 데이터 저장
    path_abs_faiss_index = get_vector_store_path_from_yaml(path_retriever_yaml)
    delete_faiss_index(path_abs_faiss_index)
    assert not os.path.exists(path_abs_faiss_index)

    node.vector_store.add_texts(["Hello world"])
    node.vector_store.save()

    assert os.path.exists(path_abs_faiss_index)

    # node_fn 실행 (임베딩/벡터스토어가 실제 파일에서 로드됨)
    # 검색 결과는 보통 빈 리스트이거나, 파일 설정에 따라 다를 수 있음
    output_state = node_fn({"query": "Hello world"})
    assert (
        "retrieved_docs" in output_state
    ), "output_key(retrieved_docs)가 결과 state에 존재해야 합니다."
    docs = output_state["retrieved_docs"]
    assert isinstance(docs, list), "검색 결과는 list 형태여야 합니다."
    print(f"[INFO] Retrieved docs: {docs}")
    assert len(docs) > 0
    assert docs[0].page_content == "Hello world"

    node2 = RetrieverNode.from_yaml(node_data, base_dir=base_dir)
    node_fn2 = node2.build()

    output_state2 = node_fn2({"query": "Hello world"})
    docs2 = output_state2["retrieved_docs"]
    assert docs2 == docs

    delete_faiss_index(path_abs_faiss_index)

from agentblock.embedding.embedding_factory import EmbeddingModelFactory
from agentblock.vector_store.faiss_utils import create_faiss_vector_store
from agentblock.retriever.retriever_node import RetrieverNode
from dotenv import load_dotenv
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

base_path = f"{script_dir}/"


load_dotenv()


def test_retriever_node_with_existing_yamls():
    """
    이미 물리적으로 생성된 YAML 파일 3개(embedding.yaml, vector_store.yaml, retriever.yaml)
    그리고 'faiss_index.bin'을 담을 경로를 지정해, RetrieverNode 테스트를 수행합니다.

    1) embedding.yaml로부터 Embeddings 생성
    2) 메모리 상에서 FAISS VectorStore 생성 + 텍스트 추가 + save_local(faiss_index.bin)
    3) vector_store.yaml은 path=faiss_index.bin 로 설정되어 있어야 함
    4) retriever.yaml을 통해 RetrieverNode를 불러옴
    5) query='Hello'로 검색 후, 결과를 검증
    """

    # [사용자 맞춤] 실제 파일 경로를 수정하세요.
    embedding_yaml_path = f"{base_path}/test_yaml/elements/embedding.yaml"
    retriever_yaml_path = f"{base_path}/test_yaml/retriever.yaml"
    faiss_index_path = f"{base_path}/faiss_index.bin"

    # 1) embedding.yaml에서 Embeddings 생성
    embedding_model = EmbeddingModelFactory.from_yaml(embedding_yaml_path)

    # 2) FAISS VectorStore를 in-memory로 생성 후, 테스트 텍스트를 추가
    vs = create_faiss_vector_store(embedding_model, path=None)

    vs.add_texts(["Hello world", "foo bar", "OpenAI is great"])
    # 3) FAISS 인덱스를 faiss_index.bin에 저장
    vs.save_local(faiss_index_path)

    # 4) retriever.yaml을 통해 RetrieverNode를 로드 → build()
    node = RetrieverNode.from_yaml(retriever_yaml_path)
    node_fn = node.build()

    # 5) 검색 테스트
    input_state = {"query": "Hello"}
    output_state = node_fn(input_state)

    # 검색 결과 확인
    retrieved_docs = output_state["retrieved_docs"]
    assert isinstance(retrieved_docs, list), "retrieved_docs must be a list."
    assert len(retrieved_docs) > 0, "retrieved_docs should not be empty."

    print("[INFO] Test retrieval results:", retrieved_docs)

import os
import pytest

from agentblock.embedding.embedding_reference import EmbeddingReference
from agentblock.embedding.dummy_embedding import DummyEmbedding
from langchain.embeddings import OpenAIEmbeddings
from agentblock.sample_data.tools import get_sample_data
from agentblock.tools.load_config import load_config


@pytest.fixture
def dummy_yaml_path():
    path = get_sample_data("yaml/embedding/reference/dummy_embedding.yaml")
    return path


@pytest.fixture
def openai_yaml_path():
    path = get_sample_data("yaml/embedding/reference/openai_embedding.yaml")
    return path


def load_reference_config(yaml_file_path: str):
    """
    YAML 파일을 읽고, "reference" 섹션의 첫 번째 항목을 반환
    => EmbeddingReference.from_yaml()에 넘길 dict
    """
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")

    data = load_config(yaml_file_path)
    # 예: data["reference"] = [ {...}, ... ]
    ref_list = data["references"]
    if not ref_list:
        raise ValueError(f"No 'reference' section found in {yaml_file_path}")

    # 여기서는 첫 번째 reference만 테스트한다고 가정
    return ref_list[0]


def test_dummy_embedding_reference_build(dummy_yaml_path):
    # 1) YAML 파일 로드 -> reference 설정 딕셔너리
    ref_config = load_reference_config(dummy_yaml_path)

    # 2) from_yaml
    ref_obj = EmbeddingReference.from_yaml(
        config=ref_config,
        base_dir=".",  # 경로 사용 시 필요에 따라 수정
        references_map={},  # 다른 reference가 없으므로 빈 dict
    )
    assert ref_obj.name == "dummy_emb"
    assert ref_obj.provider == "dummy"
    assert ref_obj._embedding is None

    # 3) build() -> DummyEmbedding
    emb_instance = ref_obj.build()
    assert emb_instance is not None
    assert isinstance(
        emb_instance, DummyEmbedding
    ), "Should create DummyEmbedding instance"

    # 4) embed_documents / embed_query 테스트
    docs = ["Hello", "World", "Testing"]
    vectors = emb_instance.embed_documents(docs)
    assert len(vectors) == len(docs)
    for vec in vectors:
        assert len(vec) == 5, "dimension=5"
        assert all(isinstance(x, float) for x in vec)

    query_vec = emb_instance.embed_query("One query")
    assert len(query_vec) == 5


def test_dummy_embedding_rebuild(dummy_yaml_path):
    """
    build() 재호출 시 동일 객체 캐싱 여부 검증
    """
    ref_config = load_reference_config(dummy_yaml_path)
    ref_obj = EmbeddingReference.from_yaml(ref_config, ".", {})
    emb_first = ref_obj.build()
    emb_second = ref_obj.build()
    assert (
        emb_first is emb_second
    ), "build() should return the same cached embedding instance"


def test_openai_embedding_reference_build(openai_yaml_path):
    from dotenv import load_dotenv

    load_dotenv()
    ref_config = load_reference_config(openai_yaml_path)
    ref_obj = EmbeddingReference.from_yaml(ref_config, ".", {})
    emb_instance = ref_obj.build()
    assert isinstance(emb_instance, OpenAIEmbeddings)
    # 이어서 embed_documents나 embed_query 테스트 가능

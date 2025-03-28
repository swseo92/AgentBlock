import pytest
from langchain.docstore.document import Document

from agentblock.preprocessing.langchain_wrapper.langchain_text_splitter import (
    character_text_split,
    recursive_character_text_split,
    token_text_split,
)


@pytest.fixture
def sample_documents():
    """테스트용 Document 리스트 Fixture."""
    doc1 = Document(page_content="Hello world!" * 10, metadata={"source": "doc1"})
    doc2 = Document(
        page_content="Another test document." * 10, metadata={"source": "doc2"}
    )
    return [doc1, doc2]


def test_character_text_split(sample_documents):
    """
    CharacterTextSplitter를 이용해 문서를 스플릿하는 테스트.
    chunk_size=50, chunk_overlap=10 설정으로 스플릿 후,
    전체 청크의 개수와 내용, 메타데이터를 검증합니다.
    """
    split_docs = character_text_split(
        documents=sample_documents, chunk_size=5, chunk_overlap=2, separator=" "
    )
    # chunk_size=50 이상이므로, 텍스트가 여러 청크로 나뉘어야 합니다.
    # 간단히 각 Document가 1개 이상 청크로 분리되었는지 확인
    assert len(split_docs) > len(
        sample_documents
    ), "Splitting should produce multiple chunks."
    # 메타데이터 유지 여부 확인
    for doc in split_docs:
        assert "source" in doc.metadata, "Metadata (source) should be preserved."


def test_recursive_character_text_split(sample_documents):
    """
    RecursiveCharacterTextSplitter를 이용해 문서를 스플릿하는 테스트.
    chunk_size=30, chunk_overlap=5로 설정 후,
    스플릿된 결과가 예상 범위 내인지 체크합니다.
    """
    split_docs = recursive_character_text_split(
        documents=sample_documents, chunk_size=30, chunk_overlap=5
    )
    # 기대치: chunk_size가 30이라, 문서가 여러 덩어리로 나뉨
    assert len(split_docs) > len(
        sample_documents
    ), "Should produce more chunks than input docs."
    # 메타데이터 유지 확인
    for doc in split_docs:
        assert doc.metadata.get("source") in ["doc1", "doc2"]


def test_token_text_split(sample_documents):
    """
    TokenTextSplitter를 이용해 문서를 스플릿하는 테스트.
    chunk_size=10, chunk_overlap=2로 설정 후,
    스플릿된 결과가 예상 범위 내인지 체크합니다.
    """
    split_docs = token_text_split(
        documents=sample_documents, chunk_size=10, chunk_overlap=2
    )
    # 토큰 단위 분할이 제대로 이루어지는지 확인
    # (단순히 청크 개수가 늘어나는지만 확인)
    assert len(split_docs) > len(sample_documents)
    # 메타데이터가 올바르게 유지되는지 검사
    for doc in split_docs:
        assert "source" in doc.metadata
        assert doc.metadata["source"] in ["doc1", "doc2"]

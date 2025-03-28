from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    TextSplitter,
)


def get_langchain_text_splitter(
    splitter_class, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs
) -> TextSplitter:
    """
    전달받은 TextSplitter 클래스를 생성해 반환합니다.

    :param splitter_class: 사용할 LangChain TextSplitter 클래스
                          (CharacterTextSplitter, RecursiveCharacterTextSplitter 등)
    :param chunk_size: 각 청크의 최대 길이 (기본값: 1000)
    :param chunk_overlap: 청크 간 겹치는 부분의 최대 길이 (기본값: 200)
    :param kwargs: splitter_class 초기화 시 추가로 필요한 파라미터
    :return: 초기화된 TextSplitter 객체
    """
    splitter_obj = splitter_class(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
    )
    return splitter_obj


def apply_text_split(
    splitter_obj: TextSplitter, documents: List[Document]
) -> List[Document]:
    """
    주어진 TextSplitter를 사용해 문서 리스트를 스플릿합니다.

    :param splitter_obj: 이미 생성된 TextSplitter 객체
    :param documents: LangChain Document 객체들의 리스트
    :return: 청크 단위로 스플릿된 새로운 Document 객체들의 리스트
    """
    new_docs = []
    for doc in documents:
        chunks = splitter_obj.split_text(doc.page_content)
        for chunk in chunks:
            # 분할된 텍스트를 Document로 다시 묶고,
            # 원래 Document의 metadata를 복사해 추적 가능하도록 유지
            new_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return new_docs


def character_text_split(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Document]:
    """
    CharacterTextSplitter를 이용해 문서 리스트를 스플릿합니다.

    :param documents: LangChain Document 객체 리스트
    :param chunk_size: 청크의 최대 길이
    :param chunk_overlap: 청크 간 겹치는 부분의 최대 길이
    :param kwargs: TextSplitter 생성에 필요한 추가 인자
    :return: 청크 단위로 스플릿된 Document 객체 리스트
    """
    splitter_obj = get_langchain_text_splitter(
        CharacterTextSplitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    new_docs = apply_text_split(splitter_obj, documents)
    return new_docs


def recursive_character_text_split(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Document]:
    """
    RecursiveCharacterTextSplitter를 이용해 문서 리스트를 재귀적으로 스플릿합니다.

    :param documents: LangChain Document 객체 리스트
    :param chunk_size: 청크의 최대 길이
    :param chunk_overlap: 청크 간 겹치는 부분의 최대 길이
    :param kwargs: TextSplitter 생성에 필요한 추가 인자
    :return: 청크 단위로 스플릿된 Document 객체 리스트
    """
    splitter_obj = get_langchain_text_splitter(
        RecursiveCharacterTextSplitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    new_docs = apply_text_split(splitter_obj, documents)
    return new_docs


def token_text_split(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Document]:
    """
    TokenTextSplitter 이용해 문서 리스트를 재귀적으로 스플릿합니다.

    :param documents: LangChain Document 객체 리스트
    :param chunk_size: 청크의 최대 길이
    :param chunk_overlap: 청크 간 겹치는 부분의 최대 길이
    :param kwargs: TextSplitter 생성에 필요한 추가 인자
    :return: 청크 단위로 스플릿된 Document 객체 리스트
    """
    splitter_obj = get_langchain_text_splitter(
        TokenTextSplitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
    )
    new_docs = apply_text_split(splitter_obj, documents)
    return new_docs

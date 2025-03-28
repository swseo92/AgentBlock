import os
from typing import Dict, Any, List
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader


def pdf_loader(inputs: Dict[str, Any], *args, **kwargs) -> List[Document]:
    # 기본 키를 'file_path'로 통일
    file_path = inputs.get("file_path")
    if not file_path:
        raise ValueError("Missing 'file_path' in inputs")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()  # 각 페이지를 Document로

    # 필요하면 메타데이터에 소스 추가
    for doc in docs:
        doc.metadata["source"] = file_path

    return docs

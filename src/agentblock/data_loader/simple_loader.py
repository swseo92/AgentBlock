import os
import requests
from typing import Dict, Any, List
from langchain.schema import Document


def simple_file_loader(inputs: Dict[str, Any], *args, **kwargs) -> List[Document]:
    """
    단일 텍스트 파일을 읽어 Document 리스트를 만드는 예시 구현.
    """
    file_path = inputs.get("file_path")
    if not file_path:
        raise ValueError("Missing 'file_path' in inputs")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    encoding = kwargs.get("encoding", "utf-8")
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()

    return [Document(page_content=content, metadata={"source": file_path})]


def simple_api_loader(inputs: Dict[str, Any], *args, **kwargs) -> List[Document]:
    """
    간단한 GET API 호출 -> 응답 텍스트를 Document로 변환하는 예시.
    """
    api_url = kwargs.get("api_url")
    if not api_url:
        raise ValueError("Missing 'api_url' in kwargs")

    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.text

    return [Document(page_content=data, metadata={"url": api_url})]

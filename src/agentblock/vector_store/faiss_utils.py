import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings.embeddings import Embeddings


def create_faiss_vector_store(embedding_model: Embeddings, path: str = None, **kwargs):
    """
    FAISS VectorStore를 생성하거나, 기존 인덱스를 로컬에서 로드합니다.

    - embedding_model: 임베딩 객체 (예: OpenAIEmbeddings)
    - path: 기존 인덱스 파일 경로 (None이면 새 인덱스 생성)
    - **kwargs: top_k, docstore, 기타 FAISS에 전달할 파라미터
    """
    # 먼저 임의로 "hello" 문장을 임베딩해서 차원 수를 파악
    vector_dim = len(embedding_model.embed_query("hello"))
    index = faiss.IndexFlatL2(vector_dim)

    if path is not None and os.path.exists(path):
        # 기존 인덱스를 로드하는 경우
        vector_store = FAISS.load_local(
            path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        # 새로 인덱스를 생성하는 경우
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            **kwargs,
        )

    vector_store.save = lambda: vector_store.save_local(path)
    vector_store._path_save = path

    return vector_store

from typing import Dict
from agentblock.base import BaseReference
from agentblock.vector_store.faiss_utils import create_faiss_vector_store

from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings


class VectorStoreReference(BaseReference):
    """
    FAISS VectorStore를 관리하는 비실행 노드.
    - EmbeddingReference를 내부적으로 참조하여, 임베딩 로직을 사용
    - build()를 통해 로컬 인덱스를 로드하거나, 필요 시 새로 생성할 수도 있음
    """

    def __init__(
        self,
        name: str,
        provider: str,
        config: Dict = None,
        embedding_ref: Embeddings = None,
    ):
        super().__init__(name)
        self.provider = provider  # 예: "faiss"
        self.config = config or {}
        self.embedding_ref = embedding_ref  # EmbeddingReference 객체
        self._vector_store = None  # build() 완료 후, langchain VectorStore 객체

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, "BaseReference"]
    ) -> "VectorStoreReference":
        """
        YAML 설정 + 다른 references를 기반으로 VectorStoreReference 생성
        예: reference:
               embedding: "openai_embedding"
        """
        ref_name = config["name"]
        cfg = config["config"]
        provider = cfg["provider"]

        # embedding 레퍼런스 이름 가져오기
        ref_links = cfg.get("reference", {})
        embedding_name = ref_links.get("embedding")
        embedding_ref = None
        if embedding_name:
            # references_map에서 EmbeddingReference 객체를 가져옴
            emb_obj = references_map.get(embedding_name)
            if not emb_obj:
                raise ValueError(
                    f"Embedding reference '{embedding_name}' not found in references_map."
                )
            if not isinstance(emb_obj, Embeddings):
                raise TypeError(
                    f"Reference '{embedding_name}' is not an EmbeddingReference."
                )
            embedding_ref = emb_obj

        return VectorStoreReference(
            name=ref_name,
            provider=provider,
            config=cfg,
            embedding_ref=embedding_ref,
        )

    def build(self):
        """
        실제 FAISS VectorStore를 생성/로드하여 self._vector_store에 보관
        - EmbeddingReference를 build()하여 LangChain Embeddings 객체 획득
        - param dict에서 path, 등 필요한 인자 로드
        """
        if self._vector_store is not None:
            return self._vector_store  # 캐싱

        if self.provider == "faiss":
            # 1) Embedding 준비
            if not self.embedding_ref:
                raise ValueError(
                    "FAISS requires an EmbeddingReference, but none found."
                )
            embedding_obj = self.embedding_ref  # build된 형태를 받음

            # 2) 파라미터 확인 (예: path)
            param_dict = self.config.get("param", {})
            faiss_path = param_dict.get("path")

            self._vector_store = create_faiss_vector_store(embedding_obj, faiss_path)
        else:
            raise ValueError(f"Unsupported vector store provider: {self.provider}")

        return self._vector_store

    # 선택: 편의 메서드 추가
    def add_documents(self, docs: list[str]) -> None:
        """
        단순 예시: 문자열 리스트 -> [Document]로 변환 후 벡터스토어에 삽입
        """
        if not self._vector_store:
            self.build()
        doc_objects = [Document(page_content=text) for text in docs]
        self._vector_store.add_documents(doc_objects)

    def search(self, query: str, k: int = 3):
        """
        쿼리로 검색해 top-k 문서 반환 (예: similarity_search)
        """
        if not self._vector_store:
            self.build()
        return self._vector_store.similarity_search(query, k=k)

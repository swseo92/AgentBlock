from typing import List
from langchain_core.embeddings.embeddings import Embeddings


class DummyEmbedding(Embeddings):
    """
    테스트용으로 사용하는 가짜 임베딩.
    embed_query, embed_documents에서 고정 길이 벡터를 반환하거나,
    임의의 값으로 채워넣어도 된다.
    """

    def __init__(self, dimension: int = 3):
        """
        dimension: 반환할 벡터 차원 수 (테스트 편의를 위해 기본값 3)
        """
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 여기서는 모든 텍스트에 대해 동일한 임의 벡터(혹은 0 벡터)를 반환
        return [self._dummy_vector() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._dummy_vector()

    def _dummy_vector(self) -> List[float]:
        # 단순히 0.1로 채워진 벡터를 예시로 함
        return [0.1] * self.dimension

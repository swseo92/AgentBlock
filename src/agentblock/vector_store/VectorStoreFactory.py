import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings.embeddings import Embeddings


class VectorStoreFactory:
    def __init__(self):
        pass

    def create_vector_store(
        self, provider: str = "faiss", embedding_model: Embeddings = None, **kwargs
    ):
        """
        provider와 embedding_model에 따라 적절한 벡터 스토어 인스턴스를 생성합니다.
        현재는 provider가 "faiss"일 경우 LangchainFAISSVectorStore,
        "annoy"일 경우 AnnoyVectorStore를 반환합니다.
        """
        if provider == "faiss":
            if embedding_model is None:
                raise ValueError(
                    "embedding_model must be provided for FAISS vector store."
                )
            return self.creat_faiss_vector_store(embedding_model, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    @staticmethod
    def creat_faiss_vector_store(
        embedding_model: Embeddings, path: str = None, **kwargs
    ):
        vector_dim = len(embedding_model.embed_query("hello"))
        index = faiss.IndexFlatL2(vector_dim)

        if path is not None:
            vector_store = FAISS.load_local(
                path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            vector_store = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                **kwargs,
            )

        return vector_store

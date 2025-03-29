import inspect
from typing import Dict, Any, List
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from agentblock.function.base import FunctionNode  # FunctionNode 상속


class EmbeddingNode(FunctionNode):
    def __init__(
        self,
        name: str,
        method: str,
        reference: Embeddings,
        input_keys: list,
        output_key: str,
    ):
        super().__init__(name, input_keys, output_key)
        self.method = method
        self.reference = reference
        self.input_keys = input_keys
        self.output_key = output_key

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, Any]
    ) -> "EmbeddingNode":
        input_keys = config["input_keys"]
        output_key = config["output_key"]

        cfg = config["config"]
        method = cfg["param"]["method"]
        reference_name = cfg["reference"]["embedding"]
        reference = references_map.get(reference_name)

        # 타입 체크: reference가 Embeddings 타입인지 확인
        if not isinstance(reference, Embeddings):
            raise ValueError(
                f"Reference must be an instance of Embeddings, got {type(reference)}"
            )

        return EmbeddingNode(
            name=config["name"],
            method=method,
            reference=reference,
            input_keys=input_keys,
            output_key=output_key,
        )

    def parse_config(self, config: dict, base_dir: str = None):
        """Configuration parsing logic (can be adjusted)"""
        # No additional config needed for now, but can be customized later
        pass

    def import_target_function(self):
        """Import the method to call (embed_query or other methods)"""
        if not hasattr(self.reference, self.method):
            raise ValueError(
                f"Method '{self.method}' not found in the reference object."
            )
        self._func = getattr(self.reference, self.method)

    def _get_method_signature(self):
        """Check the method signature to ensure it accepts a single string argument"""
        method_signature = inspect.signature(self._func)
        params = list(method_signature.parameters.values())
        first_param = params[0] if params else None
        return first_param.annotation if first_param else None

    def call_target_function(self, inputs: Dict[str, List[Document]]) -> List[Document]:
        # 문서 리스트 가져오기
        docs = list()
        for docs_each in inputs.values():
            docs.extend(docs_each)

        method_signature = self._get_method_signature()
        if method_signature == str:
            # method가 단일 string을 처리하는 경우
            embedding_vectors = [self._func(doc.page_content) for doc in docs]
        elif method_signature == List[str]:
            # method가 List[str]을 처리하는 경우
            embedding_vectors = self._func([doc.page_content for doc in docs])
        else:
            raise ValueError(f"Unsupported method signature: {method_signature}")
        return (docs, embedding_vectors)

from typing import Dict, Any
from agentblock.function.base import FunctionNode
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore


class DataSaverNode(FunctionNode):
    """
    DataSaverNode는 입력 데이터를 벡터 스토어에 저장하는 실행 노드입니다.
    YAML 설정에서 vector_store 레퍼런스를 받아, 해당 저장소에 문서를 추가합니다.

    예시 YAML 설정:

    nodes:
      - name: my_vector_store_saver
        type: data_saver
        input_keys:
          - documents
        output_key: result
        config:
          reference:
            vector_store: my_faiss
    """

    def __init__(
        self,
        name: str,
        reference: Any,  # 벡터 스토어 레퍼런스 객체
        input_keys: list,
        output_key: str,
    ):
        super().__init__(name, input_keys, output_key)
        self.reference = reference

    @staticmethod
    def from_yaml(
        config: dict, base_dir: str, references_map: Dict[str, Any]
    ) -> "DataSaverNode":
        """
        YAML 설정을 파싱하여 DataSaverNode 객체를 생성합니다.
        config 예시:
          config:
            reference:
              vector_store: my_faiss
        """
        input_keys = config["input_keys"]
        output_key = config["output_key"]
        cfg = config["config"]
        reference_name = cfg["reference"]["vector_store"]
        reference = references_map.get(reference_name)

        if reference is None:
            raise ValueError(f"Reference '{reference_name}' not found.")
        return DataSaverNode(
            name=config["name"],
            reference=reference,
            input_keys=input_keys,
            output_key=output_key,
        )

    def parse_config(self, config: dict, base_dir: str = None):
        # 추가 설정이 필요하지 않으므로 현재는 pass 처리합니다.
        pass

    def import_target_function(self):
        # 이 노드는 외부 Python 함수를 import할 필요가 없으므로, 별도 처리가 필요하지 않습니다.
        pass

    def call_target_function(self, inputs: Dict[str, Any]) -> Any:
        """
        입력 데이터("documents")를 벡터 스토어에 저장합니다.
        - 입력 데이터가 문자열이면 Document 객체로 변환합니다.
        - 저장 후, 저장된 문서 수와 상태 정보를 반환합니다.
        """
        # 입력에서 "documents" 키의 데이터를 가져옵니다.
        docs_input = inputs.get("documents")
        if docs_input is None:
            raise ValueError("Input 'documents' is required.")

        # 입력이 리스트가 아니라면 리스트로 변환합니다.
        if not isinstance(docs_input, list):
            docs_input = [docs_input]

        docs = []
        # 각 항목을 Document 객체로 변환합니다.
        for item in docs_input:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, str):
                docs.append(Document(page_content=item))
            else:
                raise ValueError(
                    "Each document must be a string or a Document instance."
                )

        # 벡터 스토어 인스턴스인지 확인하고 문서를 저장합니다.
        if isinstance(self.reference, VectorStore):
            self.reference.add_documents(docs)
            # 저장 후, 상태와 저장된 문서 수를 반환합니다.
            return {"status": "saved", "num_docs": len(docs)}
        else:
            raise ValueError(
                f"Reference must be an instance of VectorStore, got {type(self.reference)}"
            )

import inspect
from typing import Dict, Any, List, TypedDict, Union, Optional, Tuple
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
        # __init__ 메서드는 지원하지 않음
        if self.method == "__init__":
            raise ValueError(f"메서드 '__init__'는 지원하지 않습니다.")
            
        if not hasattr(self.reference, self.method):
            raise ValueError(
                f"메서드 '{self.method}'를 reference 객체에서 찾을 수 없습니다."
            )
        
        # 메서드 가져오기
        method = getattr(self.reference, self.method)
        
        # 바운드 메서드인지 확인
        if hasattr(method, '__self__'):
            self._func = method
        else:
            # 바운드되지 않은 경우, reference 객체에 바인딩
            self._func = method.__get__(self.reference, type(self.reference))

    def _get_method_signature(self):
        """메서드의 실제 시그니처를 확인하여 입력 타입을 반환"""
        if not hasattr(self, '_func'):
            raise ValueError("함수가 임포트되지 않았습니다. import_target_function을 먼저 호출하세요.")
            
        # 메서드의 파라미터 정보 가져오기
        sig = inspect.signature(self._func)
        params = list(sig.parameters.values())
        
        if not params:
            raise ValueError("메서드에 파라미터가 없습니다.")
            
        # 첫 번째 파라미터의 타입 힌트 확인
        param_type = params[0].annotation
        
        # Union 타입 처리
        if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
            # Union 타입 중 str, int 또는 List[str]이 있는지 확인
            for t in param_type.__args__:
                if t is str:
                    return str
                if t is int:
                    return int
                if hasattr(t, '__origin__') and t.__origin__ is list and t.__args__[0] is str:
                    return List[str]
            raise ValueError(f"지원하지 않는 Union 타입입니다: {param_type}")
            
        # List[str] 타입 처리
        if hasattr(param_type, '__origin__') and param_type.__origin__ is list:
            if param_type.__args__[0] is str:
                return List[str]
            raise ValueError(f"List 타입의 요소가 str이 아닙니다: {param_type}")
            
        # str 타입 처리
        if param_type is str:
            return str
            
        # int 타입 처리
        if param_type is int:
            return int
            
        raise ValueError(f"지원하지 않는 파라미터 타입입니다: {param_type}")

    def call_target_function(self, inputs: Dict[str, List[Document]]) -> Tuple[List[Document], List[List[float]]]:
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
        elif method_signature == int:
            # method가 int를 처리하는 경우 (예: __init__)
            # 이 경우에는 임베딩을 생성하지 않고 빈 리스트를 반환
            embedding_vectors = []
        else:
            raise ValueError(f"지원하지 않는 메서드 시그니처입니다: {method_signature}")
            
        # 튜플로 결과 반환 (docs와 embedding_vectors)
        return (docs, embedding_vectors)

    def build(self):
        """
        build()에서 서브클래스 로직을 순서대로 호출:
        1) parse_config(...)
        2) import_target_function()
        3) define node_fn(...)
        """
        self._prepare()

        def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # 1) 입력을 모으고 함수 호출
                inputs = self.get_inputs(state)
                self._validate_inputs(inputs)
                
                # 2) 함수 실행
                result = self.call_target_function(inputs)

                # 3) 결과를 dict로 변환 (FunctionResult 검증 건너뛰기)
                return self._wrap_result(result)
            except Exception as e:
                raise RuntimeError(f"Error in node {self.name}: {str(e)}") from e

        return node_fn

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI


# 팩토리 클래스: 설정에 따라 적절한 LLM 서비스를 생성할 수 있습니다.
class LLMFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_llm(provider: str = "openai", **kwargs) -> BaseLLM:
        """
        provider 인자에 따라 적절한 LLM 서비스를 생성합니다.
        현재는 'langchain'이 기본 옵션이며, 향후 다른 모델을 추가할 수 있습니다.
        """
        if provider == "openai":
            return ChatOpenAI(**kwargs)
        elif provider == "other":
            # 예시: 향후 다른 LLM 모델을 위한 구현체를 추가합니다.
            raise NotImplementedError(
                "Other LLM service provider is not implemented yet."
            )
        else:
            raise ValueError(f"Unsupported LLM service provider: {provider}")

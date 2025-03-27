```markdown
# AgentBlock

AgentBlock은 **YAML 기반 그래프(Workflow)**를 통해 LLM, Python 함수, 벡터 스토어, 라우터 등을 노드 단위로 구성·실행할 수 있는 **멀티 에이전트 프레임워크**입니다.

- **노드**(Node)와 **에지**(Edges)로 워크플로우를 정의
- 다양한 노드 유형(LLM, Function, Retriever, Embedding, VectorStore, Router, from_yaml) 지원
- **단절 노드 검증**, **조건부 분기**, **서브그래프 로딩** 등 다채로운 기능 내장

## 주요 특징

1. **YAML 선언**  
   - 프로젝트의 플로우(START→노드→END)를 YAML 파일 하나로 간단히 표현  
   - LLM, Embedding, VectorStore 등 구성 요소도 모두 “노드”로 통일

2. **조건 분기(Conditional Edges)**  
   - Router 노드를 통해 “domain=finance”면 금융 파이프라인, 아니면 일반 파이프라인 등 다양한 분기 처리

3. **from_yaml (서브그래프)**  
   - YAML 안에서 또 다른 YAML(서브그래프)을 로딩해 재귀적으로 대형 워크플로우를 구성

4. **Embedding/VectorStore** 분리  
   - Retriever가 Embedding + VectorStore를 내부적으로 참조  
   - Embedding/VectorStore는 **비실행 노드**로, 그래프 단절 검사에서 제외

5. **다형성(Polymorphism)**  
   - Retriever에서 Embedding/VectorStore를 **이름** 또는 **별도 파일**(`from_file`)로 로딩 가능

## 설치

```bash
git clone https://github.com/your-org/AgentBlock.git
cd AgentBlock
pip install -e .
```

> PyPI 배포를 진행했다면:  
> ```bash
> pip install agentblock
> ```

## 빠른 시작

아래는 **간단한 Retriever** 예시 YAML입니다(`sample_retriever.yaml`).  
FAISS VectorStore를 불러와 질의(`query`)에 대한 문서를 검색한 뒤, “retrieved_docs”를 반환합니다.

```yaml
nodes:
  - name: my_embedding
    type: embedding
    config:
      provider: openai
      model: text-embedding-ada-002

  - name: my_vector_store
    type: vector_store
    config:
      provider: faiss
      path: "faiss_index.bin"

  - name: my_retriever
    type: retriever
    input_keys: ["query"]
    output_key: "retrieved_docs"
    config:
      embedding:
        name: "my_embedding"
      vector_store:
        name: "my_vector_store"
      search_method: "invoke"
      search_type: "similarity"
      search_kwargs:
        k: 5

edges:
  - from: START
    to: my_retriever
  - from: my_retriever
    to: END
```

### Python에서 로딩·실행

```python
from agentblock.graph_builder import GraphBuilder

builder = GraphBuilder("sample_retriever.yaml")
graph = builder.build_graph()
result = graph.invoke({"query": "Hello world"})
print(result["retrieved_docs"])
```

- `graph.invoke`에 state를 넣어 호출하면, “my_retriever” 노드가 Embedding + VectorStore를 이용해 문서를 검색합니다.

## 문서

### YAML 스키마

- **모든 노드**는 `name`(고유 식별자), `type`(노드 유형), `config`(세부 설정) 필드로 구성
- “실행 노드”(LLM, Function, Retriever, Router, from_yaml) vs “비실행 노드”(Embedding, VectorStore)로 구분
- 자세한 규칙과 예시는 **[docs/yaml-schema.md](docs/yaml-schema.md)** 참고


## 라이선스

- Apache License 2.0

---

자세한 내용은 **/docs** 디렉토리를 참고해 주시기 바랍니다.  
질문이나 개선 사항은 언제든 이슈로 남겨주세요. 감사합니다!
```
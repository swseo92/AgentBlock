```markdown
# AgentBlock

‼ 현재 개발 중인 프로젝트입니다. 일부 기능이 미완성 상태일 수 있습니다.

**AgentBlock**은 **YAML 기반 그래프(Workflow)**로 LLM, Python 함수, 벡터 스토어, 라우터 등을 노드 단위로 구성·실행할 수 있는 **멀티 에이전트 프레임워크**입니다.

- **노드**(Node)와 **에지**(Edges)로 파이프라인(Workflow)을 정의  
- YAML 스키마에서 **비실행 노드**(Embedding, VectorStore 등)는 `references` 섹션, **실행 노드**(LLM, Function, Retriever 등)는 `nodes` 섹션에 선언  
- **단절 노드 검증**, **조건 분기**, **서브그래프 로딩** 등 다양한 기능

---

## 주요 특징

1. **YAML 선언**  
   - 프로젝트의 플로우(`START → 노드 → END`)를 YAML 파일로 간단히 표현  
   - LLM, Embedding, VectorStore 등 구성 요소는 “비실행 노드(`references` 섹션)” 혹은 “실행 노드(`nodes` 섹션)”로 구분

2. **조건 분기(Conditional Edges)**  
   - Router 노드나 `condition`을 통해 “domain=finance”면 금융 파이프라인, “A/B” 분기 등 다양한 조건부 실행

3. **from_yaml (서브그래프)**  
   - YAML 안에서 또 다른 YAML(서브그래프)을 로딩해, 대규모 워크플로우를 재귀적으로 구성

4. **Embedding/VectorStore 분리**  
   - Retriever 등에서 Embedding & VectorStore를 참조해 검색  
   - Embedding/VectorStore 자체는 비실행 노드로, BFS(실행 경로) 검증에서 제외

5. **파라미터 통일(`param`)**  
   - YAML `config` 내부에서 임베딩이나 벡터스토어 설정은 `param` 키로 통일 (예: `model`, `path` 등)

---

## 설치

```bash
git clone https://github.com/your-org/AgentBlock.git
cd AgentBlock
pip install -e .
```

(혹은 PyPI 배포 시:  
```bash
pip install agentblock
```  
)

---

## 빠른 시작 예시

아래는 **단순한 Retriever** 예시 YAML(`sample_retriever.yaml`).  
**references** 섹션에서 임베딩 & 벡터스토어를 정의하고, **nodes** 섹션에 리트리버 노드를 선언, `edges`로 연결합니다.

```yaml
references:
  - name: my_embedding
    type: embedding
    config:
      provider: openai
      param:
        model: text-embedding-ada-002

  - name: my_vector_store
    type: vector_store
    config:
      provider: faiss
      param:
        path: "faiss_index.bin"

nodes:
  - name: my_retriever
    type: retriever
    input_keys: ["query"]
    output_key: "retrieved_docs"
    config:
      reference:
        vector_store: "my_vector_store"
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
graph = builder.build()
result = graph.invoke({"query": "Hello world"})
print(result["retrieved_docs"])
```

- `graph.invoke`에 state를 넣어 호출하면, **my_retriever** 노드가 Embedding/VectorStore를 이용해 문서를 검색합니다.

---

## 문서

### YAML 스키마 (references + nodes + edges)

- 최상위에 **references**, **nodes**, **edges**만 허용  
- **비실행 노드**는 references에 (예: embedding, vector_store)  
- **실행 노드**는 nodes에 (예: retriever, function, llm 등), BFS 검사 대상  
- 자세한 규칙·예시는 **[docs/yaml-schema.md](docs/yaml-schema.md)** 참고

---

## 라이선스

- Apache License 2.0

---

좀 더 자세한 내용과 예시는 **/docs** 디렉토리에서 확인하세요.  
궁금한 점이나 제안 사항은 언제든 이슈(issues)로 남겨주세요. 감사합니다!
```
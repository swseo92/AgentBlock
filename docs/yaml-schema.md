```markdown
# AgentBlock YAML 스키마 개발 문서

본 문서는 **AgentBlock** 프로젝트에서 사용하는 **YAML 스키마**를 정리하여, 
Embedding/VectorStore/LLM/Function/Retriever/Router/FromYAML 등의 노드를 통일된 방식으로 정의하고, 
그래프(Edges) 검증 로직을 어떻게 처리하는지에 대해 안내합니다.

---

## 1. 개요

- **AgentBlock**은 노드(Node)와 에지(Edges)로 구성된 “그래프(Workflow)”를 선언적으로 정의하여,
  - LLM (예: OpenAI, HF),
  - Python 함수(Function),
  - 벡터스토어(VectorStore),
  - 라우터(Router)
  등 다양한 컴포넌트를 연결·실행하는 **프레임워크**입니다.

- 본 **YAML 스키마**의 목표:
  1. 모든 구성 요소(Embedding, VectorStore, LLM, Function, Retriever, Router 등)를 **통일된 Node 형태**로 정의
  2. **비실행 노드**(Embedding, VectorStore)는 그래프의 “단절 검사”에서 제외 (엣지 연결 불필요)
  3. **실행 노드**(LLM, Function, Router, Retriever, from_yaml 등)는 **START→Node→END** 흐름에 포함
  4. “from_file”를 통한 **서브 그래프**(subgraph) 로딩도 지원
  5. Retriever 등에서 **Embedding/VectorStore**를 “이름 기반” 또는 “파일 로딩(from_file)” 방식으로 다형적으로 참조할 수 있음

---

## 2. 최상위 구조

스키마 최상위는 **nodes**와 **edges**로 구성됩니다:

```yaml
nodes:
  - name: <node1>
    type: <node_type>
    ...
  - name: <node2>
    type: <node_type>
    ...

edges:
  - from: START
    to: <node1>
  - from: <node1>
    to: <node2>
    condition: ...
  ...
```

- **nodes**: 각 Node(노드)의 목록  
  - `name` (고유 식별자), `type` (노드 유형), 필요 시 `input_keys`, `output_key`, `config` 등
- **edges**: 노드 간 연결 관계  
  - “실행 노드”를 어떻게 연결할지 정의  
  - “비실행 노드”는 연결되지 않아도 에러 아님

---

## 3. Node 타입별 정의

### 3.1 공통 필드

| 필드           | 타입                  | 설명                                                                                                |
|----------------|-----------------------|-----------------------------------------------------------------------------------------------------|
| **name**       | `string` (필수)      | 노드 이름 (그래프 내에서 유일해야 함)                                                                |
| **type**       | `string` (필수)      | 노드 유형(예: `embedding`, `vector_store`, `retriever`, `llm`, `function`, `router`, `from_yaml` 등) |
| **input_keys** | `list(string)` (옵션)| 이 노드가 참조할 상태 키 목록 (그래프 실행 시 `state` 딕셔너리에서 가져옴)                          |
| **output_key** | `string or list`(옵션)| 이 노드가 실행 결과를 저장할 상태 키(들).                                                           |
| **config**     | `dict` (옵션)        | 노드별 세부 설정 파라미터                                                                            |

### 3.2 노드 유형

- **embedding**  
  - 임베딩 모델(예: OpenAIEmbeddings 등)을 정의하는 비실행 노드  
  - edges에서 연결되지 않아도 OK  
  - 예:  
    ```yaml
    - name: default_embedding
      type: embedding
      config:
        provider: openai
        model: text-embedding-ada-002
    ```

- **vector_store**  
  - FAISS, Pinecone 등 벡터 스토어를 정의하는 비실행 노드  
  - 예:  
    ```yaml
    - name: my_faiss
      type: vector_store
      config:
        provider: faiss
        path: "faiss_index.bin"
    ```

- **retriever**  
  - Embedding + VectorStore를 결합해, “query → 문서 검색”을 수행하는 실행 노드  
  - 예:  
    ```yaml
    - name: my_retriever
      type: retriever
      input_keys:
        - query
      output_key: retrieved_docs
      config:
        embedding:
          name: "default_embedding"   # or from_file: "embedding.yaml"
        vector_store:
          name: "my_faiss"           # or from_file: "vector_store.yaml"
        search_method: "invoke"
        search_type: "similarity"
        search_kwargs:
          k: 5
    ```

- **llm**  
  - OpenAI 등 LLM 모델 호출을 위한 실행 노드  
  - 예:  
    ```yaml
    - name: default_llm
      type: llm
      input_keys: ["query"]
      output_key: "answer"
      config:
        provider: openai
        kwargs:
          model_name: "gpt-3.5-turbo"
        prompt_template: |
          You are a helpful assistant...
    ```

- **function**  
  - Python 함수를 동적 임포트하여 실행하는 노드  
  - 예:  
    ```yaml
    - name: process_func
      type: function
      input_keys:
        - a
        - b
      output_key:
        - sum
        - diff
      config:
        function_path: "mypkg.module:my_func"
    ```

- **router**  
  - 상태값(예: `state["domain"]`)을 보고 ‘route’ 값을 설정, 이후 `condition`으로 분기하게 하는 실행 노드  
  - 예:  
    ```yaml
    - name: domain_router
      type: router
      input_keys: ["domain"]
      output_key: "route"
      config:
        route_map:
          finance: "finance_retriever"
          legal: "legal_retriever"
        default_route: "default_retriever"
    ```

- **from_yaml**  
  - 별도 YAML 파일(혹은 inline `graph:`)을 로드해 **서브 그래프**를 구성하는 노드  
  - 예:  
    ```yaml
    - name: sub_flow
      type: from_yaml
      from_file: "path/to/sub_graph.yaml"
    ```

---

## 4. Edges & 검증 로직

### 4.1 Edges

```yaml
edges:
  - from: START
    to: domain_router
  - from: domain_router
    to: my_retriever
    condition: "finance"
  - from: domain_router
    to: default_llm
    condition: "general"
  - from: my_retriever
    to: END
  - from: default_llm
    to: END
```

- **START**, **END**는 예약 노드 이름  
- “조건부” 분기는 `condition`을 통해, RouterNode 등에서 `route` 값을 반환 → 매칭되는 edge로 이동  
- **단 하나의 노드**만 “END”로 이어지는 edge를 가져야 하는 등, 빌드 시 검증이 이뤄짐

### 4.2 검증 로직(단절 노드 검사)

1. **모든 실행 노드**(retriever, llm, function, router, from_yaml 등)는 “START→...→END” 경로에 연결되어야 함  
2. **비실행 노드**(embedding, vector_store)는 edges와 무관해도 에러가 없음  
3. START에서 도달 불가능한 실행 노드가 있거나, 실행 노드가 END로 이어지지 못하면 에러  
4. END 노드로 가는 edge가 2개 이상이면 에러

---

## 5. 다형성(Polymorphism) - “이름 vs from_file”

**retriever** 등에서 Embedding/VectorStore를 지정할 때, `config.embedding`(혹은 `config.vector_store`) 아래에 두 가지 케이스를 허용합니다:

1. **name**  
   - 예:  
     ```yaml
     embedding:
       name: "default_embedding"
     ```
   - 이미 같은 YAML(혹은 상위 그래프)에 존재하는 노드 이름을 참조

2. **from_file**  
   - 예:  
     ```yaml
     embedding:
       from_file: "embedding_sub.yaml"
     ```
   - 별도의 YAML 파일(그 안에 embedding 노드가 정의됨)을 로드해 사용

구현 시, `"name"`과 `"from_file"`가 동시에 쓰이면 에러 처리.  
그 외 로직(예: “파일 로드해서 노드 빌드”, “이미 로드된 노드 캐시에서 가져오기”)은 **GraphBuilder**나 **RetrieverNode** 내부에 구현해둡니다.

---

## 6. 종합 예시

아래는 하나의 YAML에서, **비실행 노드(embedding/vector_store)**, **실행 노드(retriever/router)**, 그리고 `edges`를 모두 선언하고, “이름 참조” / “파일 참조”를 혼합한 모습입니다.

```yaml
nodes:
  # 1) 비실행 노드들
  - name: default_embedding
    type: embedding
    config:
      provider: openai
      model: text-embedding-ada-002

  - name: main_vector_store
    type: vector_store
    config:
      provider: faiss
      path: "faiss_index_main.bin"

  # 2) 실행 노드들
  - name: retriever_by_name
    type: retriever
    input_keys: ["query"]
    output_key: "docs_name"
    config:
      embedding:
        name: "default_embedding"
      vector_store:
        name: "main_vector_store"
      search_method: "invoke"
      search_type: "similarity"
      search_kwargs:
        k: 3

  - name: retriever_by_file
    type: retriever
    input_keys: ["query"]
    output_key: "docs_file"
    config:
      embedding:
        from_file: "embedding_sub.yaml"
      vector_store:
        from_file: "vector_store_sub.yaml"
      search_method: "invoke"
      search_type: "similarity"
      search_kwargs:
        k: 3

  - name: domain_router
    type: router
    input_keys: ["route_input"]
    output_key: "route"
    config:
      route_map:
        file: "retriever_by_file"
      default_route: "retriever_by_name"

edges:
  - from: START
    to: domain_router
  - from: domain_router
    to: retriever_by_file
    condition: "retriever_by_file"
  - from: domain_router
    to: retriever_by_name
    condition: "retriever_by_name"
  - from: retriever_by_file
    to: END
  - from: retriever_by_name
    to: END
```

- `default_embedding`, `main_vector_store`는 edges와 무관하지만, BFS 검사에서 제외.  
- `retriever_by_name` / `retriever_by_file`는 실행 노드로서 START→domain_router→...→END 흐름에 포함.  
- `retriever_by_file`에서는 “embedding_sub.yaml” / “vector_store_sub.yaml”을 로드.  
- RouterNode(`domain_router`)에 따라 “file” vs “name” 루트가 선택됨.

---

## 7. 마무리

- 이 스키마 설계를 통해 **AgentBlock**은 “모든 것을 Node로 통일”하면서, **Embedding/VectorStore** 같은 비실행 노드와 **LLM/Function/Retriever** 같은 실행 노드를 구분할 수 있습니다.  
- **edges** 검증에서는 **실행 노드**만을 대상으로 단절 검사를 수행해, START→...→END 무결성을 유지합니다.  
- **다형성**(Embedding/VectorStore - 이름 vs 파일)도 지원해, 어느 한쪽 방식을 강제하지 않고 사용자 편의를 높였습니다.  
- **from_yaml**으로 서브 그래프를 재귀 로딩하여 복잡한 워크플로우도 간단히 구성 가능합니다.

이 문서를 참고하여 팀원 간 컨벤션을 맞추고, 다양한 노드 타입을 확장(예: “pdf_loader”, “translator” 등)해 나가실 수 있습니다.
```

AgentBlock YAML 스키마 (업데이트 버전)

이 문서는 **AgentBlock** 프로젝트에서 사용하는 **YAML 스키마**를 최신화한 버전으로,  
**비실행 노드(Embedding, VectorStore 등)는 `references` 섹션**에,  
**실행 노드(Retriever, LLM, Function 등)는 `nodes` 섹션**에 선언함으로써 구조를 명료화하고 확장성을 높인 방식입니다.  
또한 **비실행 노드(Reference)가 다른 Reference를 참조**할 수 있는 패턴(예: VectorStore가 Embedding 참조)도 예시로 보여줍니다.

---

## 1. 개요

- **AgentBlock**은 노드(Node)와 에지(Edges)로 구성된 “그래프(Workflow)”를 선언적으로 정의하여,
  - LLM (예: OpenAI, HF),
  - Python 함수(Function),
  - 벡터스토어(VectorStore),
  - 라우터(Router),
  등 다양한 컴포넌트를 연결·실행하는 **프레임워크**입니다.

- 이번 스키마에서 **비실행 노드**와 **실행 노드**를 명확히 분리함으로써:
  1. 비실행 노드는 `edges` 연결 없이도 에러가 되지 않고 (BFS 검사 제외)
  2. 실행 노드는 `START→...→END` 경로의 단절 검사를 명확히 수행
  3. 실행 노드가 필요로 하는 비실행 노드를 `config.references`에서 **이름**으로 참조
  4. **레퍼런스(비실행 노드)끼리도** 내부적으로 **서로 참조**할 수 있으며, BFS 검사와 무관하게 리소스/구성 정보를 공유

이를 통해 유지보수성을 높이고, 노드가 늘어나는 상황에도 구조적 혼란을 줄일 수 있습니다.

---

## 2. 최상위 구조

새로운 스키마 최상위는 **references**, **nodes**, **edges** 3개 섹션으로 구성합니다:

```yaml
references:
  # 비실행 노드 목록 (embedding, vector_store 등)

nodes:
  # 실제 BFS 경로에 놓일 실행 노드 (retriever, function, llm, router 등)

edges:
  # 실행 노드 간 연결 정의, START→...→END
```

### 2.1 references

- **비실행 노드**(embedding, vector_store, tokenizer 등)를 선언
- BFS와 무관하므로, `edges` 연결 없이 정의해도 에러가 발생하지 않음
- **reference 안에서 reference**를 가질 수도 있음 (예: VectorStore가 Embedding을 참조)

### 2.2 nodes

- **실행 노드**(retriever, llm, function, router, from_yaml 등)
- `input_keys`, `output_key`, `config` 등을 정의
- `config.references` 필드로, 필요한 비실행 노드를 **이름**으로 참조

### 2.3 edges

- 실행 노드 간의 흐름(“START→node1→node2→END”)을 정의
- 단절 노드, 다중 END 등을 검사

---

## 3. 비실행 노드 vs 실행 노드

### 3.1 비실행 노드 (references 섹션)

#### 예시 1: Embedding
```yaml
- name: my_embedding
  type: embedding
  config:
    provider: openai
    model: text-embedding-ada-002
```
- edges로 연결되지 않아도 OK
- 실행 노드가 `references.embedding: "my_embedding"`으로 참조 가능

#### 예시 2: VectorStore (내부에서 Embedding을 참조)
```yaml
- name: my_vector_store
  type: vector_store
  config:
    provider: faiss
    path: "faiss_index.bin"
    reference:
      embedding: "my_embedding"
```
- 마찬가지로 edges 연관 없이 저장
- **reference** 키를 통해 **my_embedding**을 내부적으로 참조 (BFS와 무관)

여기서 VectorStore가 Embedding을 사용해 인덱스를 구성하거나 검색 로직을 수행할 때, `embedding: "my_embedding"`을 통해 연결됩니다.

그 밖에 PDF Loader, Tokenizer, API Client 등도 동일한 방식으로 **references**에 추가할 수 있습니다.

### 3.2 실행 노드 (nodes 섹션)

#### 예시: Retriever (VectorStore만 참조)
```yaml
- name: my_retriever
  type: retriever
  input_keys:
    - query
  output_key: retrieved_docs
  config:
    references:
      vector_store: "my_vector_store"
    search_method: "invoke"
    search_type: "similarity"
    search_kwargs:
      k: 5
```
- 실제 BFS에 참여하는 **실행 노드**로서, “`query` → 유사 문서” 검색을 담당
- `config.references` 안에 벡터스토어 이름을 명시(`my_vector_store`)
- 벡터스토어 내부에서 이미 “embedding”을 참조하고 있을 수 있음

#### 예시: LLM
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

#### 예시: Function
```yaml
- name: embed_and_store
  type: function
  input_keys: ["split_docs"]
  output_key: "store_result"
  config:
    function_path: "my_module:embed_and_store_func"
    references:
      embedding: "my_embedding"
      vector_store: "my_vector_store"
```

---

## 4. Edges (실행 노드 흐름)

```yaml
edges:
  - from: START
    to: text_splitter

  - from: text_splitter
    to: embed_and_store

  - from: embed_and_store
    to: END
```

- **START**, **END**는 예약 노드 이름
- RouterNode가 있을 경우 `condition` 필드로 분기 처리 가능
- 검증 시:
  1. 실행 노드가 모두 “START→…→END” 경로에 포함돼야 함
  2. END 노드로의 중복 경로가 없어야 함

---

## 5. 전체 예시

아래는 **references**(비실행 노드)와 **nodes**(실행 노드), 그리고 **edges**를 종합한 예시입니다:

```yaml
references:
  - name: my_embedding
    type: embedding
    config:
      provider: openai
      model: text-embedding-ada-002

  - name: my_vector_store
    type: vector_store
    config:
      provider: faiss
      path: "faiss_index_main.bin"
      reference:
        embedding: "my_embedding"

nodes:
  - name: text_splitter
    type: function
    input_keys: ["raw_docs"]
    output_key: "split_docs"
    config:
      function_path: "langchain_text_splitter:character_text_split"
      chunk_size: 500
      chunk_overlap: 50

  - name: embed_and_store
    type: function
    input_keys: ["split_docs"]
    output_key: "store_result"
    config:
      function_path: "my_module:embed_and_store_func"
      references:
        embedding: "my_embedding"
        vector_store: "my_vector_store"

edges:
  - from: START
    to: text_splitter
  - from: text_splitter
    to: embed_and_store
  - from: embed_and_store
    to: END
```

1. **비실행 노드**(`my_embedding`, `my_vector_store`)는 `references` 섹션에 있음.  
   - `my_vector_store` 내부에서 `embedding: "my_embedding"`을 참조  
2. **실행 노드**(`text_splitter`, `embed_and_store`)는 `nodes` 섹션에 있음.  
3. `text_splitter`와 `embed_and_store`를 edges로 연결해 BFS 수행.  
4. `embed_and_store` 노드는 `config.references.embedding` = `"my_embedding"`, `vector_store` = `"my_vector_store"` 등을 통해 비실행 노드를 참조

---

## 6. 검증 로직 & 구현 포인트

1. **GraphBuilder**  
   - `references` 섹션을 먼저 파싱해 비실행 노드를 `reference_map["my_embedding"] = EmbeddingObj...` 등으로 생성  
   - “VectorStore → Embedding”처럼, **reference 안에 또 reference**가 있는 경우에도 순서를 맞춰 빌드  
   - `nodes` 섹션을 파싱해 실행 노드를 빌드  
   - `edges`를 사용해 BFS 경로 구성 → 실행 노드만 단절 검사

2. **비실행 노드**  
   - edges 연결 없음  
   - 필요 시 “from_file” 방식(별도 YAML)도 지원 가능  
   - 내부적으로 다른 reference(예: vector_store가 embedding을 참조) 가능

3. **실행 노드에서 references**  
   - 예:
     ```yaml
     config:
       references:
         embedding: "my_embedding"
         vector_store: "my_vector_store"
     ```
   - 빌드 시점 또는 실행 시점에 `reference_map["my_embedding"]` / `reference_map["my_vector_store"]`를 찾아 객체를 주입

4. **다형성**  
   - “name vs from_file”  
   - “embedding” vs “vector_store” vs “tokenizer” 등 다양하게 확장 가능

---

## 7. 결론

- 새 스키마에서 **비실행 노드**를 `references` 섹션으로 분리하면:
  1. 구조가 명확해져, 실행 노드들의 BFS 검사 로직이 단순화  
  2. 비실행 노드가 늘어나도 유지보수성이 올라감  
  3. 실행 노드에서 간결하게 name으로 참조 가능  
  4. 레퍼런스 간에도 자유롭게 참조(예: VectorStore → Embedding) 가능

- **references** + **nodes** + **edges** 구성을 통해,  
  - **가독성**과 **확장성**을 동시에 확보할 수 있으며,  
  - “from_file” 등 기존 확장 기능과의 병행도 문제없음

이 문서를 토대로 팀 컨벤션을 맞추고, 다양한 노드 유형을 계속 확장해 보시기 바랍니다.  
특히 “reference 안에 reference”를 허용하면, **VectorStore, Loader** 등이 **Embedding, Tokenizer**를 자연스럽게 재사용할 수 있어, 대규모 파이프라인에서도 유연한 구성을 유지할 수 있습니다.
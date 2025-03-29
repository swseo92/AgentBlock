문서에 `input_keys`와 함수 인자 간의 매핑에 대한 정보를 추가하여 업데이트하였습니다. 매핑 규칙에 대한 내용을 명확하게 설명하는 섹션을 추가하는 방식으로 적용했습니다.

### 업데이트된 YAML 스키마 문서:
```markdown
# AgentBlock YAML 스키마 (최신 버전)

이 문서는 **AgentBlock** 프로젝트에서 사용하는 **YAML 스키마**를 최신화한 버전으로,  
**비실행 노드(Embedding, VectorStore 등)는 `references` 섹션**에,  
**실행 노드(Retriever, LLM, Function 등)는 `nodes` 섹션**에 선언함으로써 구조를 명료화하고 확장성을 높인 방식입니다.

또한 **비실행 노드(Reference)가 다른 Reference를 참조**할 수도 있으며(예: VectorStore가 Embedding 참조),  
**최상위에는 references, nodes, edges 3개 섹션**만 허용하고, **노드/레퍼런스의 config** 안에서는 **`param`** 키를 사용하도록 통일합니다.

---

## 1. 개요

- **AgentBlock**은 노드(Node)와 에지(Edges)로 구성된 “그래프(Workflow)”를 선언적으로 정의하여,
  - LLM (예: OpenAI, HF),
  - Python 함수(Function),
  - 벡터스토어(VectorStore),
  - 라우터(Router)
  등 다양한 컴포넌트를 연결·실행하는 **프레임워크**입니다.

- 이번 스키마에서 다음과 같은 변화가 있었습니다:
  1. **비실행 노드**와 **실행 노드**를 `references` / `nodes`로 분리  
  2. **최상위 YAML에는 `references`, `nodes`, `edges`만 존재**  
  3. **비실행 노드끼리도** 내부적으로 참조 가능 (예: VectorStore → Embedding)  
  4. **config 내부에서는 “param” 키**로 통일해 부가 설정을 넣도록 함 (예: `param: { model: "...", ... }`)

이를 통해 유지보수성을 높이고, 노드가 늘어나는 상황에도 구조적 혼란을 줄일 수 있습니다.

---

## 2. 최상위 구조

새로운 스키마의 최상위는 **references**, **nodes**, **edges** 3개 섹션만 허용하며, 다른 키가 있으면 에러가 납니다. 예:

```yaml
references:
  # 비실행 노드 목록 (embedding, vector_store 등)

nodes:
  # 실행 노드 목록 (retriever, function, llm 등)

edges:
  # 노드 간 흐름 정의, START→...→END
```

### 2.1 references

- **비실행 노드**(embedding, vector_store, tokenizer 등)  
- BFS와 무관하므로, edges와 연결 없이 정의해도 에러 없음  
- **서로를 참조**할 수도 있음(예: VectorStore → Embedding)

### 2.2 nodes

- **실행 노드**(retriever, llm, function, router, from_yaml 등)  
- `input_keys`, `output_key`, `config` 내에 설정을 담음  
- config에서 **reference**(단수) 키를 사용해 다른 비실행 노드 이름을 참조할 수 있음  
- 필요 시 “param”을 사용해 추가 설정을 넣을 수도 있으나(LLM 등), “params”는 허용되지 않음

### 2.3 edges

- 노드 간의 흐름(“START→node1→node2→END”)  
- BFS 검사에서 단절 노드, 중복 END 등 체크

---

## 3. 비실행 노드 vs 실행 노드

### 3.1 비실행 노드 (references 섹션)

#### 예시: Embedding

```yaml
- name: my_embedding
  type: embedding
  config:
    provider: openai
    param:
      model: text-embedding-ada-002
```

- edges로 연결되지 않아도 OK  
- 실행 노드가 “embedding: my_embedding” 등을 통해 참조

#### 예시: VectorStore (Embedding 참조)

```yaml
- name: my_vector_store
  type: vector_store
  config:
    provider: faiss
    param:
      path: "faiss_index_main.bin"
    reference:
      embedding: "my_embedding"
```

- 비실행 노드, BFS 연결 없음  
- 내부 `reference` 키를 통해 “my_embedding”을 참조  
- “param” 키(복수 형태 “params”는 에러 처리)로 path 등 설정

### 3.2 실행 노드 (nodes 섹션)

#### 예시: Retriever (VectorStore 참조)

```yaml
- name: my_retriever
  type: retriever
  input_keys:
    - query
  output_key: retrieved_docs
  config:
    reference:
      vector_store: "my_vector_store"
    search_method: "invoke"
    search_type: "similarity"
    search_kwargs:
      k: 5
```

- BFS에서 “`query` → 유사 문서” 검색  
- 노드가 vector_store를 참조

#### 예시: LLM

```yaml
- name: default_llm
  type: llm
  input_keys: ["query"]
  output_key: "answer"
  config:
    provider: openai
    param:
      model_name: gpt-3.5-turbo
      temperature: 0.3
    prompt_template: |
      You are a helpful assistant...
```

- BFS에서 한 번 호출될 LLM 노드  
- “param” 키로 model_name, temperature 등 설정

#### 예시: Function (Embedding & VectorStore 참조)

```yaml
- name: embed_and_store
  type: function
  input_keys: ["split_docs"]
  output_key: "store_result"
  config:
    function_path: "my_module:embed_and_store_func"
    reference:
      embedding: "my_embedding"
      vector_store: "my_vector_store"
```

- Python 함수 노드, BFS에서 “split_docs”를 받아서 임베딩 + 저장

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

- “START” / “END”는 예약어  
- RouterNode가 있다면 `condition` 필드로 분기 가능  
- 중복 END, 단절 노드 검사 시 BFS로 확인

---

## 5. 예시 종합

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
      reference:
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

1. 비실행 노드(`my_embedding`, `my_vector_store`)는 references 섹션  
2. 실행 노드(`text_splitter`, `embed_and_store`)는 nodes 섹션  
3. Edge로 “START → text_splitter → embed_and_store → END” 흐름  
4. 각 config에서 “param” 키만 허용(“params” 키는 에러)  
5. VectorStore가 Embedding을 참조할 수도 있고, 실행 노드가 VectorStore/Embedding을 참조할 수도 있음

---

## 6. 검증 로직 & 구현

1. **최상위**  
   - “references”, “nodes”, “edges” 외의 필드가 있으면 에러  
   - references, nodes, edges는 모두 list인지 확인
2. **references**  
   - 각 항목이 “type”이 비실행 노드인지(embedding, vector_store 등)  
   - config 내부에서 “param”만 허용(“params”는 에러)
3. **nodes**  
   - 각 항목이 “type”이 실행 노드인지(retriever, function, llm 등)  
   - config 내부에 “param” 사용 가능, “params” 사용 시 에러  
   - reference(단수) 키로 다른 비실행 노드 이름을 명시 가능
4. **edges**  
   - “from” / “to”가 START, END 또는 존재하는 노드명인지  
   - 중복 END, 단절 노드 검사
5. **토폴로지 정렬**  
   - references끼리도 의존성(embedding → vector_store) 있다면, 그래프 빌드 시 순서 정렬로 해결

---

## 7. 결론

- **참조(Reference) vs 실행(Node)**를 명확히 분리하면 BFS 검사 로직이 단순해지고, 비실행 노드가 늘어나도 유지보수성이 좋아집니다.  
- **config** 내부에서 **“param”** 키만 허용해 설정을 통일하면, “params” 혼용으로 인한 혼동을 방지할 수 있습니다.  
- 이 스키마는 대규모 파이프라인에서도 노드 간 의존관계를 직관적으로 표현하고, VectorStore나 Embedding 같은 비실행 노드의 재사용을 용이하게 해 줍니다.

---

## 8. 매핑된 입력 키

`input_keys` 내에서 함수 인자와 매핑되는 관계를 정의할 수 있습니다. 이 매핑은 함수 호출 시 `input_keys`에 설정된 이름과 실제 함수 인자 이름을 연결하는 데 사용됩니다.

### 예시:
```yaml
nodes:
  - name: text_splitter
    type: function_from_library
    input_keys:
      - documents -> pdf_documents  # YAML에서 입력 키와 실제 함수 인자 간의 매핑
    output_key: split_docs
    config:
      from_library: "agentblock.preprocessing.text_splitter:character_text_split"
      chunk_size: 500
      chunk_overlap: 50
```

위 예시에서 `input_keys`의 `"documents -> pdf_documents"`는 `documents`를 `pdf_documents`라는 함수의 인자 이름으로 매핑합니다. 이 방식은 매핑된 `input_key`를 통해 함수 인자를 유연하게 연결할 수 있게 해 줍니다.
```

### 변경 사항 요약:
- **매핑된 입력 키**에 관한 섹션을 문서에 추가하여 `input_keys`와 함수 인자 간의 매핑 방식을 설명했습니다.
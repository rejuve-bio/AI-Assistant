# Postman Test Guide: Individual Agent Tests & Context Sharing

## Setup (All Requests)

| Setting | Value |
| :--- | :--- |
| **Method** | `POST` |
| **URL** | `http://localhost:5000/query` |
| **Body Type** | `form-data` |
| **Header** | `Authorization: Bearer <your_token>` |

> All requests use `form-data` body fields as listed in each test below.

---

## Part 1: Individual Agent Tests

Each test below is designed to trigger **one specific agent only**.

---

### Test 1 — `CalculatorTool`

> Triggers the Python REPL calculator. Explicit math keywords guarantee routing here.

| Field | Value |
| :--- | :--- |
| `question` | `What is the square root of 2048, and what is 15% of 340?` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: CalculatorTool`

---

### Test 2 — `rag_search`

> Asks about Rejuve-specific content that only exists in `sample_data.json`.

| Field | Value |
| :--- | :--- |
| `question` | `What is the Rejuve Biotech whitepaper about and what is the StemCell100 supplement?` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: rag_search` and the response contains details about the Rejuve whitepaper or StemCell100.

---

### Test 3 — `annotation_graph`

> Names a specific well-known gene — the strongest trigger for a structured graph lookup.

| Field | Value |
| :--- | :--- |
| `question` | `What are the known protein interactions for the BRCA1 gene?` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: annotation_graph` called before any other tool.

---

### Test 4 — `hypothesis_generation`

> Explicit hypothesis request with an rsID and tissue type — the only reliable trigger for this pipeline.

| Field | Value |
| :--- | :--- |
| `question` | `Generate a hypothesis for the genetic variant rs9939609 in adipose tissue.` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: hypothesis_generation` and the response contains a mechanistic explanation.

---

### Test 5 — `biogpt_search` (Last Resort)

> Asks something specific enough that `annotation_graph` and `rag_search` will both return no useful results, forcing BioGPT as the final fallback.

| Field | Value |
| :--- | :--- |
| `question` | `What is the current clinical trial status for rapamycin as a longevity intervention in humans?` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `annotation_graph` tried → `rag_search` tried → `biogpt_search` called last.

> ⚠️ **Note:** If `annotation_graph` or `rag_search` returns a partial result, the orchestrator may not fall through to BioGPT. This is expected and correct behavior.

---

## Part 2: Context Sharing (Memory) Tests

Send these as **two separate, sequential requests** using the **same `user_id`**.  
The orchestrator should use `memory_write` after the first call and `memory_read` before the second.

---

### Step 1 — Seed a Fact into Memory

| Field | Value |
| :--- | :--- |
| `question` | `What chromosome is the BRCA1 gene located on? Remember this for follow-up questions.` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: memory_write` storing something like `chromosome_of_BRCA1: chr17`.

---

### Step 2 — Follow-Up Using Stored Memory

Send this **immediately after Step 1**, in the same session:

| Field | Value |
| :--- | :--- |
| `question` | `Given what you found earlier about the BRCA1 gene, what other genes are located on the same chromosome?` |
| `context` | `{"resource": "orchestrator"}` |

**✅ Pass Condition:** Docker logs show `Action: memory_read` retrieving the stored chromosome value — **without** making a new `annotation_graph` call to re-lookup BRCA1.

> The response should reference chromosome 17 directly, proving context was reused from memory.

---



# AI Assistant Backend API

## Overview  
This is the backend API for the RejuveBio Platform AI Assistant. This AI Assistant is a specialized backend system designed to empower biological research through three core capabilities:  
1. **Knowledge Graph Querying**: Structured exploration of biological entities (like genes, proteins, transcripts) and their relationships.
2. **Document Intelligence**: Semantic search and analysis of PDFs/research papers via Retrieval-Augmented Generation (RAG).  
3. **Context-Aware Conversations**: Multi-agent dialogue management with persistent memory for continuous interactions.  

The system integrates Large Language Models (LLMs) with domain-specific validation rules to ensure biologically accurate responses.  

## Folder Structure
```
AI-ASSISTANT/
├── app/
│   ├── annotation_graph/            # Core biological knowledge graph operations
│   │   ├── annotated_graph.py       # Main graph processor: query validation/execution
│   │   ├── dfs_handler.py           # Depth-first search for biological pathfinding
│   │   ├── neo4j_handler.py         # Neo4j connection pool & Cypher query executor
│   │   └── schema_handler.py        # BioCypher schema loader/validator
│   │
│   ├── lib/
│   │   └── auth.py                  # JWT authentication decorator for API endpoints  
│   ├── llm_handle/
│   │   └── ilm_models.py            # LLM interface
|   |
│   ├── prompts/                     # LLM prompt templates
│   │
│   ├── rag/
│   │   └── rag.py                   # RAG pipeline
│   │
│   ├── storage/
│   │   └── qdrant.py                # Qdrant vector DB
│   │
│   ├── __init__.py                  # Python package initializer
│   ├── main.py                      # Core application logic & multi-agent orchestration
│   ├── memory_layer.py              # Conversation memory manager
│   ├── routes.py                    # Flask API endpoint definitions
│   └── summarizer.py                # Summarization engine
|
├── config/                          # Schema/configuration files
|
├── helper/
│   └── access_token_generator.py    # JWT token generation utility
|
├── .env.example                     # Environment variable template
├── .gitignore                      
├── docker-compose.yml               # Qdrant service definitions
├── Dockerfile                       # Application containerization setup
├── poetry.lock                      
├── pyproject.toml                   # Poetry dependencies
├── README.md                       
├── run.py                           # Flask server entry point
└── sample_data.json                
```
---
## Core Functionalities  

### 1. **Biological Knowledge Graph Engine**  
**Purpose**: Query structured biological data with schema-enforced accuracy.  

#### Key Processes:  
- Query Parsing & Validation  
  - *Entity Extraction*
  - *Schema Compliance Check*
    
<!-- - **Graph Traversal & Pathfinding**
  - Employs **Depth-First Search (DFS)** to discover biological pathways between entities.  
  - Converts raw paths into structured JSON using predefined templates (`dfs_json_format.py`).   -->

- Neo4j Integration  
- Integration with the Annotation Service


### 2. **Document Intelligence Pipeline**  
**Purpose**: Transform unstructured text/PDFs into searchable biological knowledge.  

#### Workflow:  
1. **PDF Processing**  
   - *Text Extraction*  
   - *Summarization*  

2. **Adaptive Chunking**  
3. **Semantic Search**  
   - Vector Embeddings  
   - Hybrid Storage  
     - *General Knowledge* and *User-Specific Data*  
   - Contextual Retrieval


### 3. **Conversational Intelligence**  
**Purpose**: Enable continuous, context-rich dialogues for complex research inquiries.  

#### Components:  
- **Multi-Agent Workflow** (AutoGen):  
  - `Graph Agent`  
  - `RAG Agent`

- **Memory Management**:  
  - Short-Term Context
  - Long-Term Memory

---
## System Architecture  

### **Component Mapping**  
| Folder/File | Responsibility |  
|-------------|----------------|  
| **`annotation_graph/`** | Knowledge graph operations (query validation, Neo4j interactions, graph data retrival) |  
| **`rag/`** | PDF processing, vector search, and RAG response generation |  
| **`prompts/`** | Prompts for LLM interactions |  
| **`storage/qdrant.py`** | Vector database operations |  
| **`memory_layer.py`** | Conversation history storage/retrieval |  

### **Key Technologies**  
| Component              | Technology Stack         |  
|-------------------------|--------------------------|  
| **Knowledge Graph**     | Neo4j + BioCypher        |  
| **Vector Database**     | Qdrant                   |  
| **LLM Integration**     | OpenAI + Gemini          |  
| **API Framework**       | Flask                    |  
| **Conversation Agents** | AutoGen                  |  


### **Data Flow**  
1. **User Query** → JWT Validation → Query Type Detection (Graph/RAG)  
2. **Graph Query Path**:  Entity Extraction → Schema Validation → Annotation service → Summarization  
3. **Document Query Path**:  PDF/Text Processing → Vectorization → Qdrant Search → LLM Synthesis  

---
## Program Execution  

### Startup Process  
1. **Configuration Initialization**  
   - Loads BioCypher schemas from `schema_config.yaml`.  
   - Initializes LLM providers (OpenAI/Gemini) using API keys from `.env`.  

2. **Database Connections**  
   - Establishes Neo4j connection pool with credentials from `.env`.  
   - Initializes Qdrant collections (`VECTOR_COLLECTION`, `USER_COLLECTION`).  

3. **API Server Launch**  
   - Starts Flask server with rate limiting (200 requests/minute).  
   - Registers authentication middleware for JWT validation.
---
## Authentication & Security  
- **JWT Tokens**: Required for all API endpoints. Generated via `access_token_generator.py`.
- **Rate Limiting**: Prevents API abuse (200 requests/minute/IP).  
- **Data Isolation**: User-specific PDFs/vectors stored in separate Qdrant collections.  
---
## Installation & Configuration  

### Prerequisites
- **Python 3.10+**
- **Poetry** (dependency management)
- **Docker** (for containerized deployment)
- **Docker Compose** (for multi-container orchestration)

### 1. Local Installation  

```bash
# Clone repository
git clone https://github.com/rejuve-bio/AI-Assistant.git
cd AI-Assistant

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Configure environment
cp .env.example .env
nano .env  # Update with your credentials
```

#### Required Environment Variables (`.env`)

Ensure that the environment variables are set correctly in `.env` before running the application:
* **LLM Model Configuration:**
  * `BASIC_LLM_PROVIDER`: Choose the provider for lighter tasks (openai or gemini).
  * `BASIC_LLM_VERSION`: Version for the basic model (gpt-3.5-turbo, gemini-lite, etc.).
  * `ADVANCED_LLM_PROVIDER`: Choose the provider for advanced tasks (openai or gemini).
  * `ADVANCED_LLM_VERSION`: Version for the advanced model (gpt-4o, gemini-pro, etc.).
* **API Keys:**
  * `OPENAI_API_KEY`: Your OpenAI API key.
  * `GEMINI_API_KEY`: Your Gemini API key.
* **Neo4j Configuration:**
  * `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Connection details for the Neo4j database.
* **Annotation Service Configuration:**
  * `ANNOTATION_AUTH_TOKEN`: Authentication token for the annotation service.
  * `ANNOTATION_SERVICE_URL`: The URL for the annotation service, which processes queries.

#### qdrant setup

```bash
docker run -d \
    -p 6333:6333 \
    -v qdrant_data:/qdrant/storage qdrant/qdrant
```
---
## Usage  

### 1. Authentication Setup  
Generate access token:  
```bash
python helper/access_token_generator.py
```

Include token in requests:  
```bash
-H "Authorization: Bearer your_generated_token"
```

### 2. Starting the Service  

#### Local Execution:  
```bash
python run.py
```

#### Docker Execution:  
```bash
docker-compose up --build
```

#### POST `/query`  

**Example Request**:
**using curl:**  
```bash
curl -X POST http://localhost:5002/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What enhancers are involved in the formation of the protein p78504?"}'
```
**Request Body:**

```json
{
  "query": "Your natural language query here"
}
```
**Response:**

A JSON object containing the processed results from the AI assistant, based on the model's analysis.

### 4. PDF Processing  
```bash
curl -X POST http://localhost:5002/query \
  -F "file=@research_paper.pdf" \
  -H "Authorization: Bearer your_token"
```

---
## Usage

Once your environment is configured, you can run the app and use the AI Assistant API.

### 1. Start the application:

```bash
docker-compose up --build
```

**Example using curl:**
```bash
curl -X POST http://localhost:5002/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your query here"}'
```

### 2. To stop the services, use:
  ```bash
  docker-compose down
  ```

---

## Acknowledgments  
* OpenAI for providing the GPT models.
* Google for the Gemini models.
* Neo4j for the graph database technology.
* Flask for the lightweight web framework.
* Autogen for multi-agent system

## Impact  
Enables to:  
- Query biological relationships with schema-enforced accuracy.  
- Correlate structured knowledge with unstructured research papers.  
- Maintain continuous, context-rich dialogues for complex investigations.  
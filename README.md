# AI Assistant Backend API

This is the backend API for the RejuveBio Platform AI Assistant.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **Docker** (for running the application)

## Dependency management when installing locally
* **Poetry** 

## Installation and Setup

### 1. Clone the repository

First, clone the repository and navigate to the project folder:
```bash
git clone https://github.com/rejuve-bio/AI-Assistant.git
cd AI-Assistant
```

### 2. Setting up .env files

Ensure that the environment variables are set correctly in `.env` before running the application:

* **LLM Model Configuration:**
  * `BASIC_LLM_PROVIDER`: Choose the provider for lighter tasks (openai or gemini).
  * `BASIC_LLM_VERSION`: Version for the basic model (gpt-3.5-turbo, gemini-lite, etc.).
  * `ADVANCED_LLM_PROVIDER`: Choose the provider for advanced tasks (openai or gemini).
  * `ADVANCED_LLM_VERSION`: Version for the advanced model (gpt-4o, gemini-pro, etc.).
* **API Keys:**
  * `OPENAI_API_KEY`: Your OpenAI API key.
  * `GEMINI_API_KEY`: Your Gemini API key.
* **LangSmith (optional tracing):**
  * `LANGCHAIN_TRACING_V2`: Set to `true` to enable tracing.
  * `LANGCHAIN_API_KEY`: Your LangSmith API key.
  * `LANGCHAIN_PROJECT`: LangSmith project name.
  * `LANGCHAIN_ENDPOINT`: LangSmith API endpoint.
  * `LANGCHAIN_HIDE_INPUTS`: Optional. Set to `true` to hide inputs.
  * `LANGCHAIN_HIDE_OUTPUTS`: Optional. Set to `true` to hide outputs.
* **Neo4j Configuration:**
  * `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Connection details for the Neo4j database.
* **Annotation Service Configuration:**
  * `ANNOTATION_AUTH_TOKEN`: Authentication token for the annotation service.
  * `ANNOTATION_SERVICE_URL`: The URL for the annotation service, which processes queries.
* **Redis configuration:**
  * REDIS_URL: URL to connect to Redis (e.g., redis://<REDIS_HOST>:<REDIS_PORT>/0)
  * REDIS_HOST: Redis host (e.g., localhost)
  * REDIS_PORT: Redis port (e.g., 6379)
* **Qdrant configuration:**
  * `QDRANT_CLIENT`: qdrant Port
* **MongoDB configuration:**
  * MONGO_USERNAME: MongoDB username
  * MONGO_PASSWORD: MongoDB password
  * MONGO_DATABASE: MongoDB database name
  * MONGO_URL: MongoDB connection URL (e.g., mongodb://<MONGO_USERNAME>:<MONGO_PASSWORD>@<MONGO_HOST>:<MONGO_PORT>/)

### 3. Start the application:

Copy the example environment file and fill in your actual values:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:


### 3. Start the application
```bash
docker-compose up --build
```


## Usage

### Authentication

First, generate and copy your authentication token. From the AI-Assistant project directory, run:
```bash
python helper/access_token_generator.py
```

Use this token in your API requests:
- For Postman: Add header `Authorization: Bearer your_token_here`
- For cURL: Add `-H "Authorization: Bearer your_token_here"`

### Sending a Query

**Endpoint:** `POST /query`

**Example using curl:**
```bash
curl -X POST http://localhost:5002/query \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What enhancers are involved in the formation of the protein p78504?"
```

**Response:**
A JSON object containing the processed results from the AI assistant, based on the model's analysis.


### Stopping the application

To stop the services, use:
```bash
docker-compose down
```

Ensure that the environment variables are set correctly in `.env` before running the application:

* **LLM Model Configuration:**
  * `BASIC_LLM_PROVIDER`: Choose the provider for lighter tasks (openai or gemini).
  * `BASIC_LLM_VERSION`: Version for the basic model (gpt-3.5-turbo, gemini-lite, etc.).
  * `ADVANCED_LLM_PROVIDER`: Choose the provider for advanced tasks (openai or gemini).
  * `ADVANCED_LLM_VERSION`: Version for the advanced model (gpt-4o, gemini-pro, etc.).
* **API Keys:**
  * `OPENAI_API_KEY`: Your OpenAI API key.
  * `GEMINI_API_KEY`: Your Gemini API key.
* **LangSmith (optional tracing):**
  * `LANGCHAIN_TRACING_V2`: Set to `true` to enable tracing.
  * `LANGCHAIN_API_KEY`: Your LangSmith API key.
  * `LANGCHAIN_PROJECT`: LangSmith project name.
  * `LANGCHAIN_ENDPOINT`: LangSmith API endpoint.
  * `LANGCHAIN_HIDE_INPUTS`: Optional. Set to `true` to hide inputs.
  * `LANGCHAIN_HIDE_OUTPUTS`: Optional. Set to `true` to hide outputs.
* **Neo4j Configuration:**
  * `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Connection details for the Neo4j database.
* **Annotation Service Configuration:**
  * `ANNOTATION_AUTH_TOKEN`: Authentication token for the annotation service.
  * `ANNOTATION_SERVICE_URL`: The URL for the annotation service, which processes queries.
* **Flask Configuration:**
  * `FLASK_PORT`: Port for the Flask server (default: 5002).
* **Qdrant configuration:**
  * `QDRANT_CLIENT`: Port for qdrant client(http://localhost:6333)
* **Redis configuration:**
  * REDIS_URL: URL to connect to Redis (e.g., redis://<REDIS_HOST>:<REDIS_PORT>/0)
  * REDIS_HOST: Redis host (e.g., localhost)
  * REDIS_PORT: Redis port (e.g., 6379)

* **MongoDB configuration:**
  * MONGO_USERNAME: MongoDB username
  * MONGO_PASSWORD: MongoDB password
  * MONGO_DATABASE: MongoDB database name
  * MONGO_URL: MongoDB connection URL (e.g., mongodb://<MONGO_USERNAME>:<MONGO_PASSWORD>@<MONGO_HOST>:<MONGO_PORT>/)

## 4. Pulling images used from docker hub

Once your environment is configured, setup other images we use from docker hub .

make sure you set up qdrant local client :
```bash
# Run Qdrant locally
docker run -d \
    -p 6333:6333 \
    -v qdrant_data:/qdrant/storage \
    qdrant/qdrant

# Run Redis locally
docker run -d \
    -p 6379:6379 \
    redis:6-alpine

# Run MongoDB locally
docker run -d \
    -p 27017:27017 \
    -e MONGO_INITDB_ROOT_USERNAME=admin_user \
    -e MONGO_INITDB_ROOT_PASSWORD=secure_password \
    -e MONGO_INITDB_DATABASE=app_database \
    -v mongodb_data:/data/db \
    mongo:6.0
```

### 5. Start the Flask Server
Run the Flask server with the following command:

```bash
python run.py
```
This will start the server at http://localhost:5002.


### 6. Send a POST request to the `/query` endpoint
### Authentication
First, generate and copy your authentication token:
```bash
python helper/access_token_generator.py
```
Use this token in your API requests:
- For Postman: Add header `Authorization: Bearer your_token_here`
- For cURL: Add `-H "Authorization: Bearer your_token_here"`

You can send a POST request to the `/query` endpoint to interact with the AI Assistant.

**Example using curl:**

```bash
curl -X POST http://localhost:5002/query \
  -H "Authorization: Bearer your_token_here" \
  -F "query=What enhancers are involved in the formation of the protein P78504?"
```

**Request Body:**

A form-data field:

query  =  "Your natural language query here"

**Response:**

A JSON object containing the processed results from the AI assistant, based on the model's analysis.

## Acknowledgments

* OpenAI for providing the GPT models.
* Google for the Gemini models.
* Neo4j for the graph database technology.
* Flask for the lightweight web framework.

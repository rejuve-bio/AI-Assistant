# AI Assistant Backend API

This is the backend API for the RejuveBio Platform AI Assistant.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **Poetry** (for managing dependencies)

## Installation

### 1. Clone the repository
First, clone the repository and navigate to the project folder:

```bash
git clone [https://github.com/rejuve-bio/AI-Assistant.git](https://github.com/rejuve-bio/AI-Assistant.git)
cd ai-assistant
```

### 2. Install dependencies using Poetry
Install the required dependencies for the project:

```bash
poetry install
```

### 3. Activate the virtual environment
Activate the Poetry-managed virtual environment:

```bash
poetry shell
```

## 4. Configuration
The application uses environment variables to set up its parameters.

**Environment Variables**
The `.env` file contains sensitive information like API keys, credentials, and configuration overrides. The `.env.example` file is provided as a template. You can copy it to a `.env` file and fill in your actual values.

```bash
cp .env.example .env
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
* **Neo4j Configuration:**
  * `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Connection details for the Neo4j database.
* **Annotation Service Configuration:**
  * `ANNOTATION_AUTH_TOKEN`: Authentication token for the annotation service.
  * `ANNOTATION_SERVICE_URL`: The URL for the annotation service, which processes queries.
* **Flask Configuration:**
  * `FLASK_PORT`: Port for the Flask server (default: 5001).

## Usage

Once your environment is configured, you can run the Flask server and use the AI Assistant API.

### 1. Start the Flask Server
Run the Flask server with the following command:

```bash
python run.py
```
This will start the server at http://localhost:5001.

### 2. Send a POST request to the `/query` endpoint
You can send a POST request to the `/query` endpoint to interact with the AI Assistant.

**Example using curl:**

```bash
curl -X POST http://localhost:5001/query \
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

## Acknowledgments

* OpenAI for providing the GPT models.
* Google for the Gemini models.
* Neo4j for the graph database technology.
* Flask for the lightweight web framework.

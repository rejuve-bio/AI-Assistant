# AI Assistant
backend API

## Prerequisites

- Python 3.8+
- Poetry

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rejuve-bio/AI-Assistant.git
   cd ai-assistant
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Install the required packages:
   ```
   poetry shell
   ```

## Configuration

The application uses both a YAML configuration file and environment variables:

- `config/config.yaml`: Contains settings for the LLM model, API endpoints, and file paths.
- `.env`: Contains sensitive information like API keys and can override some YAML settings.
   ```
   # depending on the chosen model in the config file
   GEMINI_API_KEY = gemini_api_key 
   OPENAI_API_KEY = openai_api_key
   ```
## Usage

1. Start the Flask server:
   ```
   python run.py
   ```

2. Send a POST request to the `/query` endpoint:
   ```
   curl -X POST http://localhost:5001/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What enhancers are involved in the formation of the protein p78504?"}'
   ```

## API Reference

### POST /query

Process a natural language query and return the results.

**Request Body:**
```json
{
  "query": "Your natural language query here"
}
```

**Response:**
A JSON object containing the processed results from the AI assistant.

## Acknowledgments

- OpenAI for the GPT model
- Google for the Gemini model

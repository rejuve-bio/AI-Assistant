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

### 2. Configure environment variables

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


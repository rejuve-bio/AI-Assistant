# Clinical Trial Matcher Agent

## Overview
The **Clinical Trial Matcher Agent** is a specialized module within the Rejuve Biotech AI Assistant. It is designed to query real-time clinical trial registries (specifically ClinicalTrials.gov) to find active, recruiting, or completed trials related to genetic variants, specific diseases, drug targets,or interventions.

## Purpose and Value Proposition
At Rejuve Biotech, the core mission is to accelerate longevity research by turning static biological data into actionable clinical insights. The Clinical Trial Matcher Agent directly serves this mission by:

1. **Bridging the Gap Between Data (mainly hypothesis Generated) and Clinical Application**: When researchers discover a potential longevity target (like the SIRT1 gene or a new aging pathway) or a compound (like Rapamycin), this agent allows them to instantly check if there are ongoing human trials for those interventions.
2. **Empowering the AI Assistant**: By providing the Orchestrator with real-time access to global trial registries, the AI Assistant transforms from a static knowledge base into a dynamic research tool that can validate hypotheses against real-world clinical activity.
3. **Accelerating Discovery**: It saves researchers hours of manual searching by instantly summarizing the most relevant trials, their phases, statuses, and specific conditions.

## How the Orchestrator Uses It
The Clinical Trial Matcher is integrated as a dedicated tool (`TrialMatcherTool`) within the AI Assistant's central **Orchestrator**. 

Because the Orchestrator acts as the "brain," it intelligently delegates tasks. If a user asks a question about human studies, clinical applications, or specific drugs, the Orchestrator autonomously decides to call the Clinical Trial Matcher Agent. 

The agent connects to the ClinicalTrials.gov API, retrieves the relevant studies, and returns a cleanly formatted summary (including Trial ID, Title, Status, Phase, and Conditions). The Orchestrator then uses this data to formulate a comprehensive answer for the user.

## Testing and Execution

### Running the AI Assistant
Ensure you have the required dependencies (including `requests` and `sacremoses`) installed. Start the system using Docker:
```bash
docker compose up --build -d
```

### How to Test via Postman
You can test the agent by sending a POST request to the Orchestrator's unified query endpoint.

1. **Endpoint**: `POST http://localhost:5000/query`  
   *(Note: Replace your server address and port if different)*
2. **Headers**:
   - `Authorization`: `Bearer <Your_Auth_Token>`
3. **Form Data (Body)**:
   - `query`: *(Enter one of the examples below)*
   - `user_id`: `<Your_User_ID>`
   - `resource` : orchestrator

### Example Test Questions
Try asking the Orchestrator the following questions to see the Clinical Trial Matcher in action:

- **Specific Drug Inquiry**:
  > *"Are there any active clinical trials for Rapamycin?"*

- **Broad Longevity Inquiry**:
  > *"Are there any active clinical trials for longevity?"*



The Orchestrator will recognize the intent, call the Clinical Trial Matcher Agent, and return a summarized list of the requested clinical trials.

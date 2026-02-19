import os
from dotenv import load_dotenv
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()

def mock_agent(input_data):
    return f"Processed: {input_data}"

def verify_langsmith():
    print("Verifying LangSmith Integration...")
    
    # Check env vars
    print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"LANGCHAIN_ENDPOINT: {os.getenv('LANGCHAIN_ENDPOINT')}")
    print(f"LANGCHAIN_API_KEY set: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
    
    if os.getenv("LANGCHAIN_TRACING_V2") != "true":
        print("[FAIL] LangSmith tracing is NOT enabled in environment variables.")
        return

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[FAIL] LANGCHAIN_API_KEY is not set.")
        return

    # Create a simple chain
    chain = RunnableLambda(mock_agent)
    
    print("\nRunning a test trace...")
    try:
        # We don't strictly need tracing_v2_enabled context manager if env vars are set,
        # but it's good to force it for the test if something is off globally.
        # However, relying on env vars is the goal here.
        
        result = chain.invoke("Hello LangSmith")
        print(f"Result: {result}")
        print("\n[SUCCESS] Test run completed.")
        print("Please check your LangSmith project 'rejuv-ai-assistant' for a new trace.")
        
    except Exception as e:
        print(f"\n[FAIL] detailed error: {e}")

if __name__ == "__main__":
    verify_langsmith()

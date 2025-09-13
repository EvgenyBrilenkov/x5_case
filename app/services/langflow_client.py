import os
import requests

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://langflow:7860")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")  # положи в .env
API_KEY = os.getenv("LANGFLOW_API_KEY")  # если включены API keys

def run_flow(prompt: str):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    body = {
        "input_value": prompt,
        "input_type": "chat",
        "output_type": "chat",
        "output_component": "chat_output",  # или ID узла вывода из JSON
        "session_id": "backend-1",
        "tweaks": {}
    }
    r = requests.post(f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}?stream=false",
                      json=body, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

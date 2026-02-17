from mistralai import Mistral
from dotenv import load_dotenv
import os
import pathlib

env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# Set your API key in the MISTRAL_API_KEY environment variable
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("Missing MISTRAL_API_KEY in environment variables.")
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant analyzing custom data."},
        {"role": "user", "content": "Hello! Can you help me analyze a CSV file later?"}
    ]
)

print(chat_response.choices[0].message.content)
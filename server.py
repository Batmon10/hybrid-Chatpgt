from flask import Flask, request, jsonify
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import Ollama
from llama_index.service_context import ServiceContext
from dotenv import load_dotenv
import openai
import os

# === Load Keys from .env ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === App Setup ===
app = Flask(__name__)
llm = Ollama(model="mistral")
service_context = ServiceContext.from_defaults(llm=llm)
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index.as_query_engine()

# === OpenAI Key Setup ===
openai.api_key = OPENAI_API_KEY

@app.route("/ask", methods=["POST"])
def ask():
    user_key = request.headers.get("x-api-key")
    if user_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_input = data.get("question")
    use_openai = data.get("use_openai", False)

    if use_openai:
        try:
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful, smart, and friendly assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            content = chat_response.choices[0].message["content"]
            return jsonify({"response": content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Use local model with prompt tuning
    instruction = (
        "You are a helpful, smart, and friendly assistant like ChatGPT. "
        "Always give clear, concise answers and be encouraging. "
        "Now answer the following question:
"
    )
    response = query_engine.query(instruction + user_input)
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

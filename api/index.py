import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

app = Flask(__name__, template_folder='../templates')

# Mengambil API Key dari Environment Variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY belum dikonfigurasi di Environment Variable."}), 500

    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    try:
        # Inisialisasi LLM Groq via LangChain
        # Model bisa disesuaikan, misalnya: llama-3.1-70b-versatile atau llama3-8b-8192
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.7
        )

        # Mengirim pesan ke LLM
        messages = [
            SystemMessage(content="Anda adalah asisten AI yang ramah, sopan, dan membantu."),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        return jsonify({"response": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel membutuhkan objek 'app' ini untuk diekspor
if __name__ == "__main__":
    app.run(debug=True)

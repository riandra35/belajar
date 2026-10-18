import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage

app = Flask(__name__, template_folder='../templates')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Dictionary untuk menyimpan memori percakapan sementara
# Catatan: Pada Vercel (serverless), variabel global ini bisa reset jika server idle. 
# Untuk level produksi, sebaiknya simpan history ini di Database (seperti Redis atau MongoDB).
chat_histories = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY belum dikonfigurasi di Environment Variable."}), 500

    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default_session") # Identifikasi user

    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="openai/gpt-oss-20b", # Pastikan model yang dipakai valid di Groq
            temperature=0.7
        )

        # 1. Inisialisasi memori jika session_id belum ada
        if session_id not in chat_histories:
            chat_histories[session_id] = [
                SystemMessage(content="Anda adalah asisten AI yang ramah, sopan, dan membantu.")
            ]

        # 2. Tambahkan pertanyaan user (HumanMessage) ke dalam riwayat
        chat_histories[session_id].append(HumanMessage(content=user_message))
        
        # 3. Kirim seluruh riwayat pesan ke LLM
        response = llm.invoke(chat_histories[session_id])
        
        # 4. Tambahkan jawaban LLM (AIMessage) ke dalam riwayat agar diingat untuk pertanyaan berikutnya
        chat_histories[session_id].append(AIMessage(content=response.content))

        return jsonify({"response": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import json
import os
import secrets
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16)

# Create OpenAI client (new syntax)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_menu_data(restaurant_name):
    filename = f"{restaurant_name}.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@app.route("/")
def index():
    restaurant_name = request.args.get("restaurant")
    if not restaurant_name:
        return "Please specify a restaurant in the URL, e.g., ?restaurant=joes_grill", 400

    menu_data = load_menu_data(restaurant_name)
    if not menu_data:
        return f"Menu for restaurant '{restaurant_name}' not found.", 404

    return render_template("index.html", restaurant_name=restaurant_name)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    restaurant_name = request.json.get("restaurant") or request.args.get("restaurant")

    if not restaurant_name:
        return jsonify({"reply": "Restaurant not specified in request."}), 400

    menu_data = load_menu_data(restaurant_name)
    if not menu_data:
        return jsonify({"reply": f"Menu for restaurant '{restaurant_name}' not found."}), 404

    if "history" not in session:
        session["history"] = []

    system_prompt = f"""
You are a helpful restaurant assistant for {menu_data['restaurant_name']}.
Rules:
1. For general questions about the menu (e.g., "What's on the menu?"), give short, friendly, conversational summaries without listing ingredients or allergens for every item.
2. For specific questions about a menu item (e.g., "What's in the Grilled Salmon?" or "Any allergens in the Caesar Salad?"), provide full details including ingredients, allergens, and dietary notes.
3. Only reference menu items from this menu JSON; do NOT invent items.
4. Track which menu item each user question refers to. If not specified, assume the last-mentioned dish.
5. Follow clarifications from the user (e.g., "I meant the salad") to apply to the context of the previous question.
6. Answers should be concise, clear, and conversational.

Menu data: {json.dumps(menu_data['menu_items'])}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session["history"])
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error contacting OpenAI API: {str(e)}"

    session["history"].append({"role": "user", "content": user_input})
    session["history"].append({"role": "assistant", "content": reply})
    session.modified = True

    return jsonify({"reply": reply})

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)
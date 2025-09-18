from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import json
import os
import secrets
from dotenv import load_dotenv

# Load variables from .env for local dev; in production, Render provides env vars
load_dotenv()

app = Flask(__name__)

# Secret key for Flask sessions (random per restart if not provided)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16)

# OpenAI client (modern SDK)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Menu loading --------
def load_menu_data(restaurant_name: str):
    """
    Load menu JSON from /menus/<slug>.json using an absolute path.
    Returns dict or None if not found/invalid.
    """
    path = os.path.join(BASE_DIR, "menus", f"{restaurant_name}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"error": f"Menu file for '{restaurant_name}' is not valid JSON."}

# -------- Routes --------
@app.route("/")
def index():
    restaurant_name = request.args.get("restaurant")
    if not restaurant_name:
        return "Please specify a restaurant in the URL, e.g., ?restaurant=joes_grill", 400

    menu_data = load_menu_data(restaurant_name)
    if not menu_data:
        return f"Menu for restaurant '{restaurant_name}' not found.", 404
    if isinstance(menu_data, dict) and menu_data.get("error"):
        return menu_data["error"], 500

    return render_template("index.html", restaurant_name=restaurant_name)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = (request.json.get("message") or "").strip()
    restaurant_name = request.json.get("restaurant") or request.args.get("restaurant")

    if not restaurant_name:
        return jsonify({"reply": "Restaurant not specified in request."}), 400

    menu_data = load_menu_data(restaurant_name)
    if not menu_data:
        return jsonify({"reply": f"Menu for restaurant '{restaurant_name}' not found."}), 404
    if isinstance(menu_data, dict) and menu_data.get("error"):
        return jsonify({"reply": menu_data["error"]}), 500

    # Initialize session chat history
    if "history" not in session:
        session["history"] = []

    # Build system prompt
    system_prompt = f"""
You are the official menu assistant for {menu_data.get('restaurant_name', restaurant_name)}.
Use ONLY the provided menu JSON to answer questions about items, ingredients, allergens, prices, and dietary tags.
If the user asks about an item not in the JSON, say it's not on this menu and suggest close matches from the menu.
If the question is ambiguous, ask one brief clarifying question.

Menu JSON (authoritative source):
{json.dumps(menu_data.get('menu_items', []))}

Rules:
- Never invent menu items or details not present in the JSON.
- For general/menu-wide questions: give a short summary and suggest relevant sections or popular items.
- For specific items: return clear ingredients, allergen flags, and dietary notes in 1–3 concise sentences.
- Keep replies under 120 words unless the user explicitly asks for more detail.
"""

    messages = [{"role": "system", "content": system_prompt}]
    # Add prior turns
    messages.extend(session["history"])
    # Add current user message
    messages.append({"role": "user", "content": user_input})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            top_p=1,
            max_tokens=350,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"Error contacting OpenAI API: {str(e)}"

    # Save to session
    session["history"].append({"role": "user", "content": user_input})
    session["history"].append({"role": "assistant", "content": reply})
    session.modified = True

    return jsonify({"reply": reply})

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(debug=True)
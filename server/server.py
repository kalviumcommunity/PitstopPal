import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/plan-trip", methods=["POST"])
def plan_trip():
    try:
        # --- 1. Get user input
        data = request.json
        start = data.get("start")
        destination = data.get("destination")
        budget = data.get("budget")
        preferences = data.get("preferences", "")

        # --- 2. RAG context (example)
        rag_context = f"""
Available stops between {start} and {destination}: Jog Falls (waterfall), Gokarna (beach), etc.
Fuel cost estimate: ₹1800
Hotel options: ₹1000-₹1200 per night
"""

        # --- 3. System prompt (dynamic prompting with rules)
        system_prompt = f"""
You are PitstopPal, an AI-powered road trip planner.

Generate a personalized travel itinerary using real-world data.
Context for trip: {rag_context}

User preferences: {preferences}
Budget: {budget}

Guidelines:
- If budget is below ₹5000, focus on affordable stays and fewer paid activities.
- If preferences include 'beaches', highlight coastal routes and water activities.
- If preferences include 'adventure', suggest trekking or outdoor experiences.
- Always output in the format: Route → Stops → Cost → Tips
"""

        # --- 4. User prompt
        user_prompt = f"""
Plan a road trip with the following details:
- Start location: {start}
- Destination: {destination}
- Budget: {budget}
- Preferences: {preferences}
"""

        # --- 5. Optional Google Search tool
        tools = [types.Tool(googleSearch=types.GoogleSearch())]

        # --- 6. Configure generation (with temperature)
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
            temperature=0.7,  # 👈 Added temperature control
            top_p=0.9, #added top p
            top_k=40 #added top k
        )

        # --- 7. Prepare contents with system + user roles
        contents = [
            types.Content(role="model", parts=[types.Part(text=system_prompt)]),
            types.Content(role="user", parts=[types.Part(text=user_prompt)])
        ]

        # --- 8. Generate itinerary from Gemini (streaming)
        plan_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                plan_text += chunk.text

        # --- 9. Fallback if streaming is empty
        if not plan_text.strip():
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=generate_content_config
            )
            plan_text = response.output_text if response.output_text else ""

        return jsonify({"plan": plan_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"plan": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

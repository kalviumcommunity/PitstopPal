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

        # --- 3. System prompt (role="model")
        #multi Shot prompting    
        system_prompt = f"""
You are PitstopPal, an AI-powered road trip planner.
Generate personalized, budget-friendly travel itineraries using real-world data.
Prioritize user preferences, suggest scenic stops, calculate estimated costs, and give travel tips.

Examples:

User Input: Start: Bangalore, Destination: Mysore, Budget: ₹5000, Preferences: Nature
Output:
Route → Bangalore → Ramanagara → Srirangapatna → Mysore
Stops → Janapada Loka, Ranganathittu Bird Sanctuary
Cost → Fuel: ₹1500, Food: ₹1200, Stay: ₹2000, Remaining: ₹300
Tips → Start early morning to avoid traffic.

User Input: Start: Delhi, Destination: Manali, Budget: ₹15000, Preferences: Adventure
Output:
Route → Delhi → Chandigarh → Kullu → Manali
Stops → Rock Garden Chandigarh, Parvati Valley
Cost → Fuel: ₹5000, Food: ₹3000, Stay: ₹6000, Remaining: ₹1000
Tips → Carry warm clothes, avoid peak weekends.

Now use the same format for this input:
Start: Hyderabad, Destination: Goa, Budget: ₹10000, Preferences: Beaches

Use this context for accuracy:
{rag_context}
Format output as: Route → Stops → Cost → Tips
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

        # --- 6. Configure generation
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )

        # --- 7. Prepare contents with system + user roles
        contents = [
            types.Content(
                role="model",  # system instructions
                parts=[types.Part(text=system_prompt)]
            ),
            types.Content(
                role="user",  # user request
                parts=[types.Part(text=user_prompt)]
            )
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
            # Use output_text instead of contents
            plan_text = response.output_text if response.output_text else ""

        return jsonify({"plan": plan_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"plan": "", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

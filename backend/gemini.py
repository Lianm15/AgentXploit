import os
from google import genai

print("GEMINI FILE LOADED FROM:", __file__)
print("API KEY IS:", os.environ.get("GOOGLE_API_KEY"))

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def run_gemini_attack(prompt: str) -> str:
    print("GEMINI FUNCTION RUNNING")

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    return response.text
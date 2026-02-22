from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()  #reads the .env file and loads GEMINI_API_KEY

#creates the connection to Gemini - one client for the whole app
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def run_gemini_attack(prompt: str) -> str:
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",  #fast and cheap model
        contents=prompt,            
        config=types.GenerateContentConfig(
            max_output_tokens=1024, #max length of Gemini's response
            temperature=1.0,        #0.0 = robotic/safe, 2.0 = unpredictable
        )
    )
    
    return response.text  #the actual text Gemini responded with
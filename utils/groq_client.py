import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

def generate_from_groq(prompt: str):
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a lab report generator."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content
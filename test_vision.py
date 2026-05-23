import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Create a tiny 1x1 base64 image
tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

models = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'qwen/qwen3-32b',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'groq/compound',
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b'
]

for model in models:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{tiny_image}",
                            },
                        },
                    ],
                }
            ]
        )
        print(f"SUCCESS: {model}")
    except Exception as e:
        print(f"FAILED {model}: {e}")

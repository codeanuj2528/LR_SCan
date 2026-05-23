import os
import base64
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from groq import Groq
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI(title="LR Scanner")

# Use absolute path for Vercel, and look for index.html directly in the main folder
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

def get_lr_number(image_bytes: bytes) -> str:
    if not client:
        return "64116_dummy"

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = "Look at the top right corner where it says 'NON NEGOTIABLE WAY BILL'. There is a number below it (e.g., 64116). Extract that exact number. Reply with ONLY the number and no other text."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    response_text = chat_completion.choices[0].message.content.strip()
    cleaned_number = re.sub(r'[^a-zA-Z0-9-]', '', response_text)
    
    if not cleaned_number:
        return "UNKNOWN_LR"
        
    return cleaned_number

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/scan")
async def scan_lr(image: UploadFile = File(...)):
    contents = await image.read()
    lr_number = get_lr_number(contents)
    
    img = Image.open(BytesIO(contents))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_pdf.name
    temp_pdf.close()
    
    img.save(pdf_path, "PDF", resolution=100.0)
    
    filename = f"{lr_number}.pdf"
    
    return FileResponse(
        path=pdf_path, 
        filename=filename, 
        media_type="application/pdf",
        background=None
    )

import os
import base64
import tempfile
import re
import logging
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from groq import Groq

# Configure logging to stdout for Vercel logs visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lr-scanner")

load_dotenv()

app = FastAPI(title="LR Scanner")

# Ensure templates directory exists using absolute path for Vercel compatibility
BASE_DIR = Path(__file__).resolve().parent

# Check both base directory and templates folder to be 100% environment-agnostic
template_dirs = [str(BASE_DIR), str(BASE_DIR / "templates")]
logger.info(f"Configuring Jinja2Templates with search paths: {template_dirs}")
templates = Jinja2Templates(directory=template_dirs)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    logger.info("Initializing Groq client with environment key.")
    client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY not found in environment. Groq client is disabled.")
    client = None

def get_lr_number(image_bytes: bytes) -> str:
    """
    Attempts to extract the waybill number using Groq Vision API.
    If any error occurs, logs it and returns a safe fallback name.
    """
    if not client:
        logger.warning("Groq client not initialized. Falling back to default name.")
        return "LR_scanned"

    try:
        # Convert bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}", exc_info=True)
        return "LR_scanned"
    
    # Prompt the vision model
    prompt = (
        "Look at the top right corner where it says 'NON NEGOTIABLE WAY BILL'. "
        "There is a number below it (e.g., 64116). Extract that exact number. "
        "Reply with ONLY the number and no other text."
    )
    
    try:
        logger.info("Sending image payload to Groq Vision API...")
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
            timeout=20.0  # Ensure we have a timeout so requests don't hang indefinitely on Vercel
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        logger.info(f"Groq API raw response: {response_text}")
        
        # Clean the response to ensure we only get valid characters (alphanumeric and dashes)
        cleaned_number = re.sub(r'[^a-zA-Z0-9-]', '', response_text)
        logger.info(f"Cleaned extracted number: {cleaned_number}")
        
        if not cleaned_number:
            logger.warning("Cleaned number is empty. Using fallback.")
            return "LR_scanned"
            
        return cleaned_number

    except Exception as e:
        logger.error(f"Groq Vision API call failed: {e}", exc_info=True)
        logger.info("Using safe fallback name 'LR_scanned' to proceed with PDF creation.")
        return "LR_scanned"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Serving home page.")
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/scan")
async def scan_lr(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    logger.info(f"Received /scan request. File: {image.filename}, Content-Type: {image.content_type}")
    
    try:
        contents = await image.read()
        if not contents:
            logger.error("Uploaded image file is empty.")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Uploaded image file is empty. Please select a valid photo."}
            )
    except Exception as e:
        logger.error(f"Failed to read uploaded image bytes: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to read image data: {str(e)}"}
        )
    
    # 1. Extract LR number with robust fallback
    lr_number = get_lr_number(contents)
    
    # 2. Open image and convert to PDF using Pillow
    try:
        logger.info("Parsing image bytes with Pillow...")
        img = Image.open(BytesIO(contents))
        
        # Convert to RGB to ensure PDF format compatibility
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        logger.error(f"Pillow image parsing failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error", 
                "message": "Invalid image file. Please upload or snap a valid image receipt (JPG, PNG, WebP)."
            }
        )
        
    # 3. Safely create temporary file in Vercel-writable /tmp directory
    # On Vercel, the only writable area is /tmp. On other platforms or locally, standard temp is fine.
    temp_dir = "/tmp" if os.path.exists("/tmp") and os.access("/tmp", os.W_OK) else None
    logger.info(f"Writing temporary files to directory: {temp_dir or 'Default System Temp'}")
    
    try:
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=temp_dir)
        pdf_path = temp_pdf.name
        temp_pdf.close()
        
        logger.info(f"Saving PDF to: {pdf_path}")
        img.save(pdf_path, "PDF", resolution=100.0)
    except Exception as e:
        logger.error(f"Failed to write PDF file to disk: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server failed to write PDF file: {str(e)}"}
        )
    
    filename = f"{lr_number}.pdf"
    logger.info(f"PDF generated successfully. Prepared download: {filename}")
    
    # Safe cleanup function to delete temp PDF from Vercel's serverless environment after download completes
    def remove_file(path: str):
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.info(f"Successfully cleaned up temp file: {path}")
        except Exception as e:
            logger.error(f"Failed to delete temp file {path}: {e}")
            
    background_tasks.add_task(remove_file, pdf_path)
    
    # Add Content-Disposition and Access-Control-Expose-Headers so frontend can read the exact filename
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition"
    }
    
    return FileResponse(
        path=pdf_path, 
        filename=filename, 
        media_type="application/pdf",
        background=background_tasks,
        headers=headers
    )

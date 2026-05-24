import os
import base64
import tempfile
import re
import logging
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from groq import Groq
import httpx

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

def upload_to_google_drive_background(
    pdf_path: str, 
    filename: str, 
    apps_script_url: str, 
    drive_folder_url: str
):
    """
    Quietly uploads the generated PDF to the user's personal Google Drive folder
    via their Google Apps Script Web App. Run as a FastAPI background task.
    """
    try:
        logger.info(f"Google Drive background upload initiated for file: {filename}")
        if not os.path.exists(pdf_path):
            logger.error("PDF path not found on disk. Cancelling Drive upload.")
            return

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        base64_data = base64.b64encode(pdf_bytes).decode("utf-8")
        
        payload = {
            "base64Data": base64_data,
            "filename": filename,
            "folderUrl": drive_folder_url
        }
        
        logger.info(f"Sending payload to Google Apps Script: {apps_script_url}")
        
        # Follow redirects is crucial for Apps Script web apps (redirects from script.google.com to googleusercontent.com)
        r = httpx.post(apps_script_url, json=payload, timeout=40.0, follow_redirects=True)
        
        if r.status_code == 200:
            logger.info(f"Google Drive upload successful! Response: {r.text}")
        else:
            logger.error(f"Google Drive upload failed with HTTP {r.status_code}: {r.text}")
            
    except Exception as e:
        logger.error(f"Unexpected error during Google Drive background upload: {e}", exc_info=True)

def get_lr_number(image_bytes: bytes) -> str:
    """
    Attempts to extract the waybill number using Groq Vision API.
    If any error occurs, logs it and returns a safe fallback name.
    """
    if not client:
        logger.warning("Groq client not initialized. Falling back to default name.")
        return "LR_scanned"

    try:
        # Load image and compress/resize to max 800px for extremely fast API processing
        logger.info(f"Compressing image for API payload. Original size: {len(image_bytes)} bytes.")
        img = Image.open(BytesIO(image_bytes))
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=70)
        api_image_bytes = buffered.getvalue()
        logger.info(f"Compressed size for Groq API: {len(api_image_bytes)} bytes.")
        
        base64_image = base64.b64encode(api_image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to compress image for API: {e}", exc_info=True)
        # Fallback to original bytes
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        except Exception:
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
async def scan_lr(
    background_tasks: BackgroundTasks, 
    image: UploadFile = File(...),
    drive_folder_url: str = Form(None),
    apps_script_url: str = Form(None)
):
    logger.info(f"Received /scan request. File: {image.filename}, Content-Type: {image.content_type}")
    
    # Log Google Drive settings status
    if apps_script_url and drive_folder_url:
        logger.info("Google Drive integration parameters received.")
    else:
        logger.info("Google Drive integration parameters not supplied.")

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
    
    # 4. Trigger Google Drive upload in the background if configured (doesn't block user download!)
    if apps_script_url and drive_folder_url:
        background_tasks.add_task(
            upload_to_google_drive_background,
            pdf_path,
            filename,
            apps_script_url,
            drive_folder_url
        )

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
    
    if apps_script_url and drive_folder_url:
        # Inform the frontend that a Google Drive backup was triggered
        headers["X-Google-Drive-Upload"] = "triggered"

    return FileResponse(
        path=pdf_path, 
        filename=filename, 
        media_type="application/pdf",
        background=background_tasks,
        headers=headers
    )

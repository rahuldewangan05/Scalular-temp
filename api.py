from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import fitz
import os
import tempfile
from dotenv import load_dotenv
from fabric_matcher import TechPackAnalyzer

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize TechPackAnalyzer
techpack_analyzer = TechPackAnalyzer()

def extract_text_from_file(file_path):
    """Extract text content from the file."""
    try:
        doc = fitz.open(file_path)
        text_content = ""
        for page in doc:
            text_content += page.get_text()
        return text_content
    except Exception as e:
        print(f"Error extracting text from file: {e}")
        return ""

@app.get("/", response_class=HTMLResponse)
@app.get("/alive", response_class=JSONResponse)
async def alive():
    return JSONResponse(content={"status": "Healthy"}, status_code=200)

@app.get('/extract_info', response_class=HTMLResponse)
async def get_file(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})

@app.post('/extract_info', response_class=JSONResponse)
async def process_file(request: Request):
    try:
        # Get form data
        form = await request.form()
        file = form.get("file")
        if not file:
            raise ValueError("No file uploaded")
        
        # Create temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save uploaded file
        file_path = os.path.join(temp_dir, 'uploaded_file')
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        
        # Extract text from file
        text_data = extract_text_from_file(file_path)
        if not text_data:
            raise ValueError("No text could be extracted from the file")
        
        # Analyze techpack using TechPackAnalyzer
        result = await techpack_analyzer.analyze_techpack(text_data)
        
        # Clean up temporary files
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        # Return default values if processing fails
        return JSONResponse(content={
            "gender": "men",
            "product_name": "",
            "zipper": False,
            "logo_embroidery": False,
            "size": "M",
            "print": "solid",
            "category": "crewneck-t-shirts",
            "quantity_in_gms": "180-200",
            "fabric_and_blend": "Single-Jersey-(Combed)"
        }, status_code=200)

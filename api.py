from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse,HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pymupdf
import google.generativeai as genai
import json
import base64 
import os
import fitz
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pdfkit
from PyPDF2 import PdfMerger
import tempfile

load_dotenv()



class quotation_response(BaseModel):
    gender: str
    quantity_in_gms: int
    product_name: str
    category: str
    zipper:bool
    logo_embroidery:bool
    size:str

class products(BaseModel):
    product:list[quotation_response]

class fabric(BaseModel):
    fabric_and_blend:str
    print:str

templates = Jinja2Templates(directory="templates")

obj=json.load(open("object.json",'r'))

app=FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")
api_key=os.environ['api_key']
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
print("HI")

@app.get("/",response_class=JSONResponse)
@app.get("/alive",response_class=JSONResponse)
async def alive():
    print("Healthy")
    return JSONResponse(content={"status":"Healthy"},status_code=200)

class File_request:
    def __init__(self,request:Request):
        self.request: Request=request
    
    async def load_data(self):
        form= await self.request.form()
        self.upload_file = form.get("file").file.read()

async def extract_info(base64_image):
    image_data = base64.b64decode(base64_image)
    
    # Create generation config for structured output
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
    }
    
    prompt = """Analyze this image and extract the following information in JSON format:
    {
        "gender": "men/women/kids",
        "quantity_in_gms": number,
        "product_name": "string",
        "category": one of ["Crewneck T-shirts","Hoodies","Polo T-shirt","Boxers","Boxer Brief","Undershirts","Boxer Shorts","Sweatshirts","Jogger Pants","Boys Top & Bottom","Women Pyjama Set","Women top & Shorts","Ladies Pyjama Set","Ladies Top & Shorts","Ladies Sleevless Top & Shorts","Women's Tank top","Women's Camisole"],
        "zipper": boolean,
        "logo_embroidery": boolean,
        "size": "2XL or 3XL"
    }"""
    
    try:
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_data}])
        print("Raw Gemini Response:", response.text)  # Debug log
        
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [data]
            else:
                print("No JSON found in response")  # Debug log
                return [{
                    "gender": "men",
                    "quantity_in_gms": 0,
                    "product_name": "",
                    "category": "Crewneck T-shirts",
                    "zipper": False,
                    "logo_embroidery": False,
                    "size": "2XL"
                }]
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")  # Debug log
            return [{
                "gender": "men",
                "quantity_in_gms": 0,
                "product_name": "",
                "category": "Crewneck T-shirts",
                "zipper": False,
                "logo_embroidery": False,
                "size": "2XL"
            }]
    except Exception as e:
        print(f"Gemini API error: {e}")  # Debug log
        raise e

def extract_fabric_from_text(file_path):
    """Extract fabric information from text content of the file."""
    fabric_keywords = [
        'fabric', 'fabric content', 'fabrication', 'fabrics',
        'design main fabric', 'style content', 'body fabric',
        'composition', 'material', 'content', 'fabric details',
        'main fabric', 'fabric specification', 'fabric type'
    ]
    
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        print(f"Total pages in document: {total_pages}")
        
        # Dictionary to store fabric sections by page
        fabric_sections_by_page = {}
        
        # Extract text from all pages
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text_content = page.get_text()
            print(f"Analyzing page {page_num + 1} of {total_pages}")
            
            # Convert to lowercase for case-insensitive matching
            text_content = text_content.lower()
            
            # Look for fabric-related sections
            page_fabric_sections = []
            for keyword in fabric_keywords:
                if keyword.lower() in text_content:
                    # Find all lines containing the keyword
                    lines = text_content.split('\n')
                    for i, line in enumerate(lines):
                        if keyword.lower() in line.lower():
                            # Get more context - look at previous and next lines
                            start_idx = max(0, i-2)
                            end_idx = min(len(lines), i+4)
                            context = ' '.join(lines[start_idx:end_idx])
                            # Clean up the context
                            context = ' '.join(context.split())
                            if context not in page_fabric_sections:  # Avoid duplicates
                                page_fabric_sections.append(context)
                                print(f"Found fabric information on page {page_num + 1}: {context[:100]}...")
            
            if page_fabric_sections:
                fabric_sections_by_page[page_num + 1] = page_fabric_sections
        
        # Combine all fabric sections
        all_fabric_sections = []
        for page_sections in fabric_sections_by_page.values():
            all_fabric_sections.extend(page_sections)
        
        print(f"Found fabric information on {len(fabric_sections_by_page)} pages")
        return all_fabric_sections
        
    except Exception as e:
        print(f"Error extracting text from file: {e}")
        return []

def find_best_fabric_match(text_sections, available_fabrics):
    """Find the best matching fabric from the available options."""
    if not text_sections:
        return None
        
    # Common fabric components and their variations
    fabric_components = {
        'cotton': ['cotton', 'ctn'],
        'polyester': ['polyester', 'poly', 'p'],
        'viscose': ['viscose', 'vis', 'rayon'],
        'spandex': ['spandex', 'elastane', 'lycra', 'sp'],
        'nylon': ['nylon', 'ny'],
        'modal': ['modal'],
        'linen': ['linen'],
        'silk': ['silk'],
        'wool': ['wool']
    }
    
    # Clean and normalize the available fabrics
    normalized_fabrics = {f.lower().replace('-', ' ').replace('_', ' '): f for f in available_fabrics}
    
    # Process each text section
    for section in text_sections:
        # Clean and normalize the text
        text = section.lower().replace('-', ' ').replace('_', ' ')
        print(f"Analyzing text section: {text[:100]}...")
        
        # First try exact match
        for norm_fabric, original_fabric in normalized_fabrics.items():
            if norm_fabric in text:
                print(f"Found exact match: {original_fabric}")
                return original_fabric
        
        # If no exact match, try to identify fabric components
        found_components = []
        for component, variations in fabric_components.items():
            if any(var in text for var in variations):
                found_components.append(component)
                print(f"Found component: {component}")
        
        if found_components:
            # Try to match with available fabrics based on components
            for norm_fabric, original_fabric in normalized_fabrics.items():
                if all(comp in norm_fabric for comp in found_components):
                    print(f"Found component-based match: {original_fabric}")
                    return original_fabric
    
    return None

async def extract_fabric(data, base64image):
    # Ensure category exists and is not None
    if not data.get('category'):
        data['category'] = 'crewneck-t-shirts'
    
    # Clean and normalize category
    data['category'] = data['category'].replace(" ", "-").lower()
    
    # Get available prints and fabric blends
    prints = ['Waterprint', "Puff Print", "HD Print", "Foil", "Sublimation", "Tie Die", 'Solid']
    vals = list(obj[data['category']].keys())
    
    # Extract fabric information from text
    temp_dir = os.path.join(os.getcwd(), 'temp')
    file_path = os.path.join(temp_dir, 'uploaded_file')
    fabric_sections = extract_fabric_from_text(file_path)
    
    # Try to find fabric match from text
    fabric_match = find_best_fabric_match(fabric_sections, vals)
    print(f"Final fabric match: {fabric_match}")
    
    # For print type, we'll use a simple image analysis
    # This could be improved with a dedicated print detection model
    print_type = 'Solid'  # Default to Solid
    
    # Set the fabric blend
    if fabric_match:
        data['fabric_and_blend'] = fabric_match
    else:
        # If no match found, use the most common fabric for the category (excluding Single-Jersey)
        category_fabrics = [f for f in vals if 'single-jersey' not in f.lower()]
        data['fabric_and_blend'] = category_fabrics[0] if category_fabrics else vals[0]
        print(f"No fabric match found, using default: {data['fabric_and_blend']}")
    
    data['print'] = print_type
    return data

def closest(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))

def correct_out(data):
    try:
        data['category'] = data['category'].replace(" ","-").lower() if data.get('category') else "crewneck-t-shirts"
        data['fabric_and_blend'] = data['fabric_and_blend'].replace(" ","-") if data.get('fabric_and_blend') else ""
        if data.get('gender', '').lower() in ("adult",'unisex'):
            data['gender'] = 'men'
        else:
            data['gender'] = data.get('gender', 'men').lower()
        
        pp = ["140-160", "180-200", "220-240", "260-280"]
        
        if data['category'] in obj:
            if data.get('fabric_and_blend') in obj[data['category']]:
                pp = list(obj[data['category']][data['fabric_and_blend']].keys())
        
        # Handle None or invalid quantity_in_gms
        try:
            val = int(data.get('quantity_in_gms', 0))
        except (TypeError, ValueError):
            print(f"Invalid quantity_in_gms value: {data.get('quantity_in_gms')}")
            val = 0
            
        new_val = None
        mins = []
        
        for x in pp:
            x = [int(j) for j in x.split("-")]
            if val > x[0]:
                if len(x) > 1:
                    if val < x[1]:
                        new_val = "-".join([str(y) for y in x])
                else:
                    new_val = "-".join([str(y) for y in x])
            mins.append(x[0])
            
        if new_val is None:
            new_val = pp[closest(mins, val)]
            
        if data.get('print') is not None:
            data['print'] = data['print'].replace(" ", "-").lower()
            
        data['quantity_in_gms'] = new_val
        
        # Ensure all required fields exist with default values
        default_values = {
            'gender': 'men',
            'quantity_in_gms': new_val,
            'product_name': '',
            'category': 'crewneck-t-shirts',
            'zipper': False,
            'logo_embroidery': False,
            'size': '2XL'
        }
        
        for key, default_value in default_values.items():
            if key not in data or data[key] is None:
                data[key] = default_value
                
        return data
    except Exception as e:
        print(f"Error in correct_out: {str(e)}")
        # Return default structure if error occurs
        return {
            'gender': 'men',
            'quantity_in_gms': '180-200',
            'product_name': '',
            'category': 'crewneck-t-shirts',
            'zipper': False,
            'logo_embroidery': False,
            'size': '2XL',
            'fabric_and_blend': '',
            'print': None
        }

@app.get('/extract_info',response_class=HTMLResponse)
async def get_file(request:Request):
    return templates.TemplateResponse(name="index.html",context={"request":request})

def encode_image(image_path=None):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def convert_excel_to_pdf(excel_path: str) -> str:
    """Convert Excel file to PDF and merge all sheets."""
    # Create a temporary directory for PDF files
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, 'output.pdf')
    
    # Read Excel file
    excel_file = pd.ExcelFile(excel_path)
    
    # Create PDF merger
    merger = PdfMerger()
    
    # Convert each sheet to PDF and merge
    for sheet_name in excel_file.sheet_names:
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Convert to HTML
        html = df.to_html()
        
        # Convert HTML to PDF
        sheet_pdf_path = os.path.join(temp_dir, f'{sheet_name}.pdf')
        pdfkit.from_string(html, sheet_pdf_path)
        
        # Add to merger
        merger.append(sheet_pdf_path)
    
    # Save merged PDF
    merger.write(pdf_path)
    merger.close()
    
    return pdf_path

@app.post('/extract_info',response_class=JSONResponse)
async def get_file(request: Request):
    try:
        form=File_request(request)
        await form.load_data()
        file=form.upload_file
        
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Save uploaded file
        file_path = os.path.join(temp_dir, 'uploaded_file')
        with open(file_path, 'wb') as f:
            f.write(file)
            
        # Check file type and handle accordingly
        if file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
            # Convert Excel to PDF
            pdf_path = convert_excel_to_pdf(file_path)
        else:
            # For PDF/AI files, use as is
            pdf_path = file_path
            
        # Convert first page to JPEG
        jpeg_path = os.path.join(temp_dir, 'tec_pack.pdf.jpeg')
        
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pixmap = page.get_pixmap()
        pixmap.set_dpi(512,512)
        img = pixmap.tobytes()
        
        with open(jpeg_path, 'wb') as f:
            f.write(img)
            f.close()
            
        item = encode_image(image_path=jpeg_path)
        resp = await extract_info(base64_image=item)
        resp = resp[0]
        resp = await extract_fabric(resp, item)
        resp = correct_out(resp)
        
        # Clean up temporary files
        try:
            os.remove(file_path)
            os.remove(jpeg_path)
            if file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
                os.remove(pdf_path)
        except:
            pass
            
        return JSONResponse(content=resp,status_code=200)
    except Exception as e:
        return JSONResponse(content=str(e),status_code=400)
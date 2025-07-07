from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
from pathlib import Path
import uuid
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.dml import MSO_THEME_COLOR_INDEX
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to Word Converter", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
CONVERTED_DIR = Path("converted")
UPLOAD_DIR.mkdir(exist_ok=True)
CONVERTED_DIR.mkdir(exist_ok=True)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

class PDFConverter:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def convert_pdf_to_docx(self, pdf_path: str, docx_path: str) -> bool:
        """
        Convert PDF to DOCX using PyMuPDF with enhanced layout preservation
        """
        try:
            doc = Document()
            pdf_document = fitz.open(pdf_path)
            
            # Add document title
            title = doc.add_heading('Converted Document', 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Add page break for subsequent pages
                if page_num > 0:
                    doc.add_page_break()
                
                # Add page heading
                page_heading = doc.add_heading(f'Page {page_num + 1}', level=1)
                page_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # Get page dimensions for image positioning
                page_rect = page.rect
                
                # Extract text with positioning information
                text_dict = page.get_text("dict")
                
                # Extract images first
                image_list = page.get_images(full=True)
                image_positions = {}
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            image_data = pix.tobytes("png")
                            image_path = os.path.join(self.temp_dir, f"image_{page_num}_{img_index}.png")
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_data)
                            
                            # Add image to document
                            paragraph = doc.add_paragraph()
                            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
                            
                            # Calculate image size (maintain aspect ratio)
                            img_width = min(6, pix.width / 100)  # Max 6 inches
                            
                            paragraph.add_run().add_picture(image_path, width=Inches(img_width))
                            
                            # Store image position for text flow
                            image_positions[img_index] = (pix.x, pix.y, pix.width, pix.height)
                        
                        pix = None
                    except Exception as img_error:
                        logger.warning(f"Could not extract image {img_index}: {img_error}")
                        continue
                
                # Process text blocks with better formatting
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        block_text = ""
                        font_sizes = []
                        is_bold = False
                        
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                                font_sizes.append(span["size"])
                                if span["flags"] & 2**4:  # Bold flag
                                    is_bold = True
                            
                            if line_text.strip():
                                block_text += line_text + "\n"
                        
                        if block_text.strip():
                            # Determine formatting based on font size
                            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                            
                            if avg_font_size > 16:
                                # Large text - likely heading
                                heading = doc.add_heading(block_text.strip(), level=2)
                                heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
                            elif avg_font_size > 14:
                                # Medium text - subheading
                                heading = doc.add_heading(block_text.strip(), level=3)
                                heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
                            else:
                                # Normal text
                                paragraph = doc.add_paragraph()
                                run = paragraph.add_run(block_text.strip())
                                
                                if is_bold:
                                    run.bold = True
                                
                                # Set font size
                                run.font.size = Pt(max(10, min(avg_font_size, 14)))
                
                # Add some spacing between pages
                doc.add_paragraph()
            
            pdf_document.close()
            doc.save(docx_path)
            return True
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return False
    
    def convert_image_to_docx(self, image_path: str, docx_path: str) -> bool:
        """
        Convert image to DOCX
        """
        try:
            doc = Document()
            doc.add_heading('Converted Image Document', 0)
            
            # Add the original image
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(image_path, width=Inches(6))
            
            # Add placeholder for OCR text
            doc.add_paragraph()
            doc.add_heading('Text Content', level=1)
            ocr_paragraph = doc.add_paragraph()
            ocr_paragraph.add_run("OCR functionality can be added here using pytesseract library.")
            ocr_paragraph.add_run("\nTo enable OCR, install: pip install pytesseract")
            
            doc.save(docx_path)
            return True
            
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    """
    Convert uploaded file (PDF/Image) to Word document
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_{file.filename}"
    output_filename = f"{unique_id}_{Path(file.filename).stem}.docx"
    
    input_path = UPLOAD_DIR / input_filename
    output_path = CONVERTED_DIR / output_filename
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize converter
        converter = PDFConverter()
        
        # Convert based on file type
        if file_ext == '.pdf':
            success = await asyncio.get_event_loop().run_in_executor(
                executor, 
                converter.convert_pdf_to_docx, 
                str(input_path), 
                str(output_path)
            )
        else:
            # Image file
            success = await asyncio.get_event_loop().run_in_executor(
                executor, 
                converter.convert_image_to_docx, 
                str(input_path), 
                str(output_path)
            )
        
        converter.cleanup()
        
        if not success:
            raise HTTPException(status_code=500, detail="Conversion failed")
        
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Conversion completed but output file not found")
        
        # Return the converted file
        return FileResponse(
            path=str(output_path),
            filename=f"{Path(file.filename).stem}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    
    finally:
        # Cleanup files after a delay
        asyncio.create_task(cleanup_files(input_path, output_path))

async def cleanup_files(input_path: Path, output_path: Path, delay: int = 300):
    """
    Clean up files after a delay (5 minutes default)
    """
    await asyncio.sleep(delay)
    try:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()
    except Exception as e:
        logger.warning(f"File cleanup failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF to Word Converter"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF to Word Conversion Service",
        "version": "1.0.0",
        "endpoints": {
            "convert": "/convert (POST)",
            "health": "/health (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
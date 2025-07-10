# Install system dependencies
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev tesseract-ocr-eng poppler-utils

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run with production settings
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
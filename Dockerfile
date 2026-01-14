# 1. Use Python 3.10 slim (lightweight and stable)
FROM python:3.10-slim

# 2. Install System Dependencies
# - poppler-utils: For pdf2image (PDF conversion)
# - libgl1 & libglib2.0-0: Required by OpenCV (used inside PaddleOCR)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory
WORKDIR /app

# 4. Copy requirements file first (to cache dependencies)
COPY requirements.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# This command will also copy the 'models' folder residing on the server
COPY . .

# 7. Expose the port (Must match the port in main.py)
EXPOSE 9000

# 8. Run the Application
CMD ["python", "main.py"]
import os
import shutil
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import openpyxl
from io import BytesIO
import uvicorn

# --- 1. IMPORT YOUR CUSTOM FUNCTION ---
# This assumes main5.py is in the same directory as main.py
# try:
from main5 import process_single_pdf_for_api
# except ImportError:
#     print("\n[CRITICAL ERROR] Could not import 'main5.py'.")
#     print("Make sure main5.py is in the same folder as this script.\n")
#     # Dummy fallback for testing if main5 isn't found
#     def process_single_pdf_for_api(path): return {}

# --- 1. App Configuration ---
app = FastAPI(title="OCR Web App")
TEMP_DIR = Path("temp_processing")
TEMP_DIR.mkdir(exist_ok=True)

# --- 2. Data Models ---
class OCRResult(BaseModel):
    file_name: str
    invoice_yn: str = Field(..., alias="invoice(Y/N)")
    visit_yn: str = Field(..., alias="Visit(Y/N)")
    transit_no: str = Field(..., alias="Transit_No")
    account_no: str = Field(..., alias="Account_No")
    prefix: str
    ilr_date: str = Field(..., alias="ILR_date")
    amount: float
    other_amount: float

    class Config:
        populate_by_name = True

# --- 3. The HTML UI (Embedded) ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>OCR App</title>
    <style>
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .controls { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        button { cursor: pointer; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; }
        .hidden-input { display: none; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        
        #status { margin-top: 10px; font-weight: bold; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; margin-right: 10px;}
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>OCR App</h1>
    
    <div class="controls">
        <h3>1. Select Documents</h3>
        <p>Choose individual files OR an entire folder.</p>
        
        <input type="file" id="fileInput" class="hidden-input" multiple accept=".pdf,.jpg,.png">
        <button onclick="document.getElementById('fileInput').click()">Select Files (Single/Multiple)</button>
        
        <span style="margin: 0 10px;">OR</span>

        <input type="file" id="folderInput" class="hidden-input" webkitdirectory directory>
        <button onclick="document.getElementById('folderInput').click()">Select Entire Folder</button>

        <div id="status"></div>
    </div>

    <div id="resultsArea" style="display:none;">
        <h3>Results</h3>
        <div class="btn-group">
            <!--<button class="json-btn" onclick="downloadJSON()">Download JSON</button>-->
            <button class="excel-btn" onclick="downloadExcel()">Download Excel</button>
        </div>
        <table id="resultTable">
            <thead>
                <tr>
                    <th>File Name</th>
                    <th>Invoice (Y/N)</th>
                    <th>Visit (Y/N)</th>
                    <th>Transit No</th>
                    <th>Account No</th>
                    <th>Prefix</th>
                    <th>Date</th>
                    <th>Amount</th>
                    <th>Other Amt</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const folderInput = document.getElementById('folderInput');
        const statusDiv = document.getElementById('status');
        let currentData = [];

        // Listeners
        fileInput.addEventListener('change', handleUpload);
        folderInput.addEventListener('change', handleUpload);

        async function handleUpload(event) {
            const files = event.target.files;
            if (files.length === 0) return;

            // UI Updates
            statusDiv.innerHTML = `<div class="loader"></div> Processing ${files.length} files...`;
            document.getElementById('resultsArea').style.display = 'none';
            const tbody = document.querySelector('#resultTable tbody');
            tbody.innerHTML = ''; // Clear old results

            // Build Form Data
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    currentData = await response.json();
                    renderTable(currentData);
                    statusDiv.innerText = `Success! Processed ${currentData.length} documents.`;
                    document.getElementById('resultsArea').style.display = 'block';
                } else {
                    statusDiv.innerText = "Error: " + response.statusText;
                }
            } catch (error) {
                statusDiv.innerText = "Error: " + error.message;
            }
            
            // Reset inputs so same file can be selected again if needed
            event.target.value = ''; 
        }

        function renderTable(data) {
            const tbody = document.querySelector('#resultTable tbody');
            data.forEach(item => {
                const row = `<tr>
                    <td>${item.file_name}</td>
                    <td>${item['invoice(Y/N)']}</td>
                    <td>${item['Visit(Y/N)']}</td>
                    <td>${item.Transit_No}</td>
                    <td>${item.Account_No}</td>
                    <td>${item.prefix}</td>
                    <td>${item.ILR_date}</td>
                    <td>${item.amount}</td>
                    <td>${item.other_amount}</td>
                </tr>`;
                tbody.innerHTML += row;
            });
        }
        
        function downloadJSON() {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentData, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "ocr_results.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }

        async function downloadExcel() {
            if(currentData.length === 0) return;
            
            // Send the current data BACK to server to convert to Excel
            try {
                const response = await fetch('/download-excel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentData)
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = "ocr_results.xlsx";
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                } else {
                    alert("Failed to generate Excel file");
                }
            } catch (e) {
                alert("Error downloading Excel: " + e.message);
            }
        }
    </script>
</body>
</html>
"""

# --- 5. Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the HTML Page"""
    return html_content

@app.post("/process", response_model=List[OCRResult])
async def process_documents(files: List[UploadFile] = File(...)):
    """API Endpoint called by the JS frontend"""
    results = []
    
    for file in files:
        # Filter for PDF/Images if folder upload picks up junk
        if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            continue

        temp_path = TEMP_DIR / Path(file.filename).name
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Run OCR Logic
            data = process_single_pdf_for_api(str(temp_path))
            results.append(data)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
        finally:
            if temp_path.exists():
                os.remove(temp_path)

    return results

# --- EXCEL GENERATION ENDPOINT ---
@app.post("/download-excel")
async def download_excel(data: List[OCRResult]):
    # 1. Create a Workbook in memory
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OCR Results"

    # 2. Add Headers
    headers = [
        "File Name", "Invoice(Y/N)", "Visit(Y/N)", 
        "Transit No", "Account No", "Prefix", 
        "ILR Date", "Amount", "Other Amount"
    ]
    ws.append(headers)

    # 3. Add Data Rows
    for item in data:
        ws.append([
            item.file_name,
            item.invoice_yn,
            item.visit_yn,
            item.transit_no,
            item.account_no,
            item.prefix,
            item.ilr_date,
            item.amount,
            item.other_amount
        ])

    # 4. Save to BytesIO stream
    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)

    # 5. Return as a file download
    headers = {
        'Content-Disposition': 'attachment; filename="ocr_results.xlsx"'
    }
    return StreamingResponse(stream, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    print("Starting Web Server...")
    print("Open http://localhost:9000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=9000)
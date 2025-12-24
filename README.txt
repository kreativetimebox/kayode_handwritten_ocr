OCR APPLICATION - DISTRIBUTION PACKAGE
======================================

This package contains a standalone OCR application for processing bank deposit forms.

CONTENTS:
---------
- main5.py                          : Main OCR application
- requirements.txt                  : Python package dependencies
- models/                           : Pre-trained OCR models (164 MB)
  - PP-OCRv5_server_det/           : Text detection model
  - PP-OCRv5_server_rec/           : Text recognition model
- install_packages.ps1              : Package installer (requires internet)
- DEPLOYMENT_INSTRUCTIONS.md        : Detailed deployment guide
- setup_models.ps1                  : Model setup helper (already configured)

PREREQUISITES:
-------------
1. Python 3.11 or 3.12 (3.13 recommended)
2. Windows 10/11 with PowerShell
3. Tesseract OCR installed at: C:\Program Files\Tesseract-OCR\
   Download from: https://github.com/UB-Mannheim/tesseract/wiki
4. Poppler installed at: C:\poppler\Library\bin\
   Download from: https://github.com/oschwartz10612/poppler-windows/releases
5. You can specify Poppler and Tesseract path in config.json

INSTALLATION STEPS:
------------------
1. Extract this ZIP file to a folder (e.g., C:\OCR_Application\)
2. Open PowerShell as Administrator
3. Navigate to the folder: cd "C:\OCR_Application"
4. Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
5. For unsigned files - unblock file: Unblock-File -Path .\install_packages.ps1
5. Run: .\install_packages.ps1
6. Wait for installation to complete (requires internet)
7. Test: .\venv\Scripts\Activate.ps1; python main5.py

QUICK START:
-----------
After installation, run the application:
1. Navigate to the folder: e.g., cd "C:\OCR_Application"
2. Activate virtual environment: .\venv\Scripts\Activate.ps1
3. Run application: python main5.py "<pdf_file_name.pdf>" -o "output_file_name.xlsx"
3. Input files by default go in: .\data\input\
4. Output Excel files appear in: .\data\ or current directory based on -o argument in step 2

TROUBLESHOOTING:
---------------
- If PowerShell script blocked: Run "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
- If Tesseract not found: Install at C:\Program Files\Tesseract-OCR\ or update line 17 in main5.py
- If Poppler not found: Install at D:\Program Files\poppler-25.07.0\ or update line 76 in main5.py
- If models not found: Verify models\ folder exists with both detection and recognition models
- If nothing works send log file from logs folder

Package Version: 2025-12-19

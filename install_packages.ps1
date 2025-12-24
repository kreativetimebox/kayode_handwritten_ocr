# install_packages.ps1
# Package Installation Script for Restricted Systems
# This script installs all required Python packages for the OCR application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OCR Application Package Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
Write-Host "[INFO] Checking Python installation..." -ForegroundColor Green
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Write-Host "[INFO] Please install Python 3.13 or compatible version" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = python --version
Write-Host "[OK] Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
Write-Host "[INFO] Checking virtual environment..." -ForegroundColor Green
if (-not (Test-Path ".\venv")) {
    Write-Host "[INFO] Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment exists" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Green
$activateScript = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "[ERROR] Activation script not found at: $activateScript" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Execute in the same session
& $activateScript
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip first
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip
Write-Host ""

# Install PaddlePaddle from official repository
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing PaddlePaddle..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[INFO] Source: https://www.paddlepaddle.org.cn/packages/stable/cpu/" -ForegroundColor Yellow
Write-Host "[INFO] This may take several minutes (100+ MB download)..." -ForegroundColor Yellow
Write-Host ""

python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to install PaddlePaddle" -ForegroundColor Red
    Write-Host "[INFO] Possible causes:" -ForegroundColor Yellow
    Write-Host "  - No internet connection" -ForegroundColor White
    Write-Host "  - Firewall blocking https://paddle-whl.bj.bcebos.com" -ForegroundColor White
    Write-Host "  - Insufficient disk space" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] PaddlePaddle installed successfully" -ForegroundColor Green
Write-Host ""

# Install remaining packages from requirements.txt
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing packages from requirements.txt..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path ".\requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Installing packages from PyPI..." -ForegroundColor Yellow
Write-Host ""

python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to install packages from requirements.txt" -ForegroundColor Red
    Write-Host "[INFO] Possible causes:" -ForegroundColor Yellow
    Write-Host "  - No internet connection" -ForegroundColor White
    Write-Host "  - Firewall blocking PyPI (https://pypi.org)" -ForegroundColor White
    Write-Host "  - Package version conflicts" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] All packages installed successfully" -ForegroundColor Green
Write-Host ""

# Verify critical packages
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$criticalPackages = @(
    "paddlepaddle",
    "paddleocr",
    "opencv-python",
    "pdf2image",
    "openpyxl",
    "pytesseract",
    "pillow",
    "numpy"
)

$allInstalled = $true

foreach ($pkg in $criticalPackages) {
    $check = python -m pip show $pkg 2>$null
    if ($LASTEXITCODE -eq 0) {
        $version = ($check | Select-String "Version:").ToString().Split(":")[1].Trim()
        Write-Host "[OK] $pkg ($version)" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $pkg" -ForegroundColor Red
        $allInstalled = $false
    }
}

Write-Host ""

if ($allInstalled) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Ensure Tesseract OCR is installed at: C:\Program Files\Tesseract-OCR\" -ForegroundColor White
    Write-Host "2. Ensure Poppler is installed at: D:\Program Files\poppler-25.07.0\Library\bin\" -ForegroundColor White
    Write-Host "3. Verify models/ folder exists with OCR models" -ForegroundColor White
    Write-Host "4. Run the application: python main5.py" -ForegroundColor White
    Write-Host ""
    Write-Host "To activate the virtual environment manually:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Installation completed with warnings!" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Some packages are missing. Please review the errors above." -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Press Enter to exit"

# install_packages_offline.ps1
# OFFLINE Package Installation Script for Restricted Systems
# Use this when internet access is completely blocked
# Prerequisites: Pre-downloaded wheel files in ./packages/ folder

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OFFLINE Package Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if packages directory exists
if (-not (Test-Path ".\packages")) {
    Write-Host "[ERROR] packages/ directory not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "For offline installation:" -ForegroundColor Yellow
    Write-Host "1. On a machine with internet, run:" -ForegroundColor White
    Write-Host "   pip download -r requirements.txt -d packages" -ForegroundColor Cyan
    Write-Host "   pip download paddlepaddle==3.2.0 -d packages -i https://www.paddlepaddle.org.cn/packages/stable/cpu/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. Copy the packages/ folder to this directory" -ForegroundColor White
    Write-Host "3. Run this script again" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python
Write-Host "[INFO] Checking Python installation..." -ForegroundColor Green
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = python --version
Write-Host "[OK] Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check/Create virtual environment
Write-Host "[INFO] Checking virtual environment..." -ForegroundColor Green
if (-not (Test-Path ".\venv")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
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
& ".\venv\Scripts\Activate.ps1"
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Count wheel files
$wheelFiles = Get-ChildItem ".\packages\*.whl" -ErrorAction SilentlyContinue
$wheelCount = ($wheelFiles | Measure-Object).Count

Write-Host "[INFO] Found $wheelCount wheel files in packages/" -ForegroundColor Green
Write-Host ""

if ($wheelCount -eq 0) {
    Write-Host "[ERROR] No wheel files found in packages/" -ForegroundColor Red
    Write-Host "[INFO] Please download packages first (see instructions above)" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install all packages from local directory
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing packages from local cache..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Install PaddlePaddle first (if exists)
$paddleWhl = Get-ChildItem ".\packages\paddlepaddle-*.whl" -ErrorAction SilentlyContinue
if ($paddleWhl) {
    Write-Host "[INFO] Installing PaddlePaddle..." -ForegroundColor Yellow
    python -m pip install --no-index --find-links=.\packages paddlepaddle==3.2.0
    Write-Host ""
}

# Install all other packages
Write-Host "[INFO] Installing remaining packages..." -ForegroundColor Yellow
python -m pip install --no-index --find-links=.\packages -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Installation failed" -ForegroundColor Red
    Write-Host "[INFO] Some packages may be missing from packages/ folder" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] All packages installed successfully" -ForegroundColor Green
Write-Host ""

# Verify installation
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
    Write-Host "OFFLINE Installation completed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Application is ready to run: python main5.py" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Installation completed with warnings!" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Some packages are missing. Please download them on a machine with internet." -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Press Enter to exit"

# download_packages.ps1
# Package Download Script (Run on machine WITH internet)
# Downloads all required Python packages for offline installation on restricted systems

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Package Download Script" -ForegroundColor Cyan
Write-Host "For Offline Installation Preparation" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

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

# Check requirements.txt
if (-not (Test-Path ".\requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create packages directory
Write-Host "[INFO] Creating packages directory..." -ForegroundColor Green
New-Item -ItemType Directory -Path ".\packages" -Force | Out-Null
Write-Host "[OK] Directory created/verified" -ForegroundColor Green
Write-Host ""

# Download PaddlePaddle
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Downloading PaddlePaddle..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[INFO] Size: ~100 MB" -ForegroundColor Yellow
Write-Host "[INFO] Source: PaddlePaddle official repository" -ForegroundColor Yellow
Write-Host ""

python -m pip download paddlepaddle==3.2.0 -d packages -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to download PaddlePaddle" -ForegroundColor Red
    Write-Host "[INFO] Check internet connection and firewall settings" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] PaddlePaddle downloaded" -ForegroundColor Green
Write-Host ""

# Download packages from requirements.txt
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Downloading packages from requirements.txt..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[INFO] This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

python -m pip download -r requirements.txt -d packages

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to download some packages" -ForegroundColor Red
    Write-Host "[INFO] Check internet connection and firewall settings" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[OK] All packages downloaded" -ForegroundColor Green
Write-Host ""

# Count downloaded files
$wheelFiles = Get-ChildItem ".\packages\*.whl"
$tarFiles = Get-ChildItem ".\packages\*.tar.gz" -ErrorAction SilentlyContinue
$totalFiles = ($wheelFiles | Measure-Object).Count + ($tarFiles | Measure-Object).Count

$totalSize = (Get-ChildItem ".\packages" -Recurse | Measure-Object -Property Length -Sum).Sum
$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Download Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total files: $totalFiles" -ForegroundColor Green
Write-Host "Total size: $totalSizeMB MB" -ForegroundColor Green
Write-Host "Location: .\packages\" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps for Restricted System:" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Copy the entire application folder to the restricted system" -ForegroundColor White
Write-Host "   (including the packages/ folder)" -ForegroundColor White
Write-Host ""
Write-Host "2. On the restricted system, run:" -ForegroundColor White
Write-Host "   .\install_packages_offline.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. The installation will use only local files (no internet needed)" -ForegroundColor White
Write-Host ""

# List key packages
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Key Packages Downloaded:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$keyPackages = @(
    "paddlepaddle",
    "paddleocr",
    "opencv-python",
    "pdf2image",
    "pytesseract",
    "openpyxl",
    "pillow",
    "numpy"
)

foreach ($pkg in $keyPackages) {
    $file = Get-ChildItem ".\packages\$pkg-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($file) {
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        Write-Host "[OK] $($file.Name) ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "[?] $pkg (check packages/ folder)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Package download completed successfully!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to exit"

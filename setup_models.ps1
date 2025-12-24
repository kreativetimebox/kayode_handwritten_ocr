# setup_models.ps1
# Helper script to copy PaddleOCR models from cache to local models folder

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PaddleOCR Model Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Define paths
$CacheBase = "$env:USERPROFILE\.paddlex\official_models"
$DetModel = "PP-OCRv5_server_det"
$RecModel = "PP-OCRv5_server_rec"
$TargetDir = ".\models"

# Check if cache directory exists
if (-not (Test-Path $CacheBase)) {
    Write-Host "[ERROR] PaddleX cache not found at: $CacheBase" -ForegroundColor Red
    Write-Host "[INFO] Please run the application once with internet to download models first" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create target directory
Write-Host "[INFO] Creating models directory..." -ForegroundColor Green
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

# Copy detection model
Write-Host "[INFO] Copying detection model: $DetModel" -ForegroundColor Green
$DetSource = Join-Path $CacheBase $DetModel
$DetTarget = Join-Path $TargetDir $DetModel

if (Test-Path $DetSource) {
    Copy-Item -Path $DetSource -Destination $DetTarget -Recurse -Force
    Write-Host "[OK] Detection model copied successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Detection model not found at: $DetSource" -ForegroundColor Red
    exit 1
}

# Copy recognition model
Write-Host "[INFO] Copying recognition model: $RecModel" -ForegroundColor Green
$RecSource = Join-Path $CacheBase $RecModel
$RecTarget = Join-Path $TargetDir $RecModel

if (Test-Path $RecSource) {
    Copy-Item -Path $RecSource -Destination $RecTarget -Recurse -Force
    Write-Host "[OK] Recognition model copied successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Recognition model not found at: $RecSource" -ForegroundColor Red
    exit 1
}

# Verify copied files
Write-Host ""
Write-Host "[INFO] Verifying model files..." -ForegroundColor Green

$RequiredFiles = @("inference.pdmodel", "inference.pdiparams", "inference.yml")

# Check detection model files
Write-Host "Checking detection model files..." -ForegroundColor Cyan
foreach ($file in $RequiredFiles) {
    $filePath = Join-Path $DetTarget $file
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length
        $sizeMB = [math]::Round($size/1MB, 2)
        Write-Host "  [OK] $file ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
    }
}

# Check recognition model files
Write-Host "Checking recognition model files..." -ForegroundColor Cyan
foreach ($file in $RequiredFiles) {
    $filePath = Join-Path $RecTarget $file
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length
        $sizeMB = [math]::Round($size/1MB, 2)
        Write-Host "  [OK] $file ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
    }
}

# Check for dictionary file
$dictFile = Join-Path $RecTarget "rec_word_dict.txt"
if (Test-Path $dictFile) {
    Write-Host "  [OK] rec_word_dict.txt" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] rec_word_dict.txt (Optional - may not be present)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Verify the models/ folder has been created" -ForegroundColor White
Write-Host "2. Package the entire application folder for distribution" -ForegroundColor White
Write-Host "3. Deploy to restricted system (no internet needed)" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"

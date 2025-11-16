# RaceIQ Setup Script for Windows PowerShell
# Toyota Gazoo Racing Hackathon 2025

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   RaceIQ - Setup and Installation" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to raceiq directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "   ! Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "   ✓ Virtual environment created" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "   ✓ Virtual environment activated" -ForegroundColor Green

Write-Host ""
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "   ✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[5/5] Running system test..." -ForegroundColor Yellow
python test_system.py

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the dashboard:" -ForegroundColor Yellow
Write-Host "  streamlit run src/ui/dashboard.py" -ForegroundColor White
Write-Host ""

# PowerShell script to launch AI Glass Optimization Tool
Write-Host "🔷 Starting AI Glass Optimization Tool..." -ForegroundColor Cyan
Write-Host "📁 Changing to application directory..." -ForegroundColor Yellow

# Change to the script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "🌐 Launching Streamlit application..." -ForegroundColor Green
Write-Host "📱 Open your browser and go to: http://localhost:8501" -ForegroundColor Magenta
Write-Host "⏹️  Press Ctrl+C to stop the application" -ForegroundColor Red
Write-Host "-" * 50 -ForegroundColor Gray

try {
    # Run the Streamlit app
    streamlit run app.py
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host "💡 Make sure you have installed the requirements:" -ForegroundColor Yellow
    Write-Host "   pip install -r requirements.txt" -ForegroundColor White
}
finally {
    Write-Host "🛑 Application stopped." -ForegroundColor Yellow
} 
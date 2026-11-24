# Start the EAGLE-DAR backend and frontend in two windows.
#   scripts\start.ps1
$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root\frontend'; npm run dev"
Write-Host "Backend:  http://127.0.0.1:8000  (Swagger: /docs)"
Write-Host "Frontend: http://localhost:5173"
Write-Host "Wait ~10 s for the models to load; the dashboard's System Online indicator turns green when ready."

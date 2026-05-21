# Stop stale Streamlit on 8501, then launch the dashboard.
$ErrorActionPreference = "SilentlyContinue"
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force }

Set-Location $PSScriptRoot
& .\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501 --server.headless false

param(
    [string]$PythonVersion = "3.12.10",
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageName = "Numerica-Portable-Windows"
$packageRoot = Join-Path $projectRoot $OutputDirectory
$stagingRoot = Join-Path $packageRoot $packageName
$runtimeZip = Join-Path $packageRoot "python-embed.zip"
$getPip = Join-Path $packageRoot "get-pip.py"
$portableZip = Join-Path $packageRoot "$packageName.zip"
$pythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"

function Download-File([string]$Url, [string]$Destination) {
    & curl.exe -L --fail --retry 3 --retry-delay 2 --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
}

if (Test-Path $packageRoot) {
    Remove-Item $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingRoot | Out-Null
$runtimeRoot = Join-Path $stagingRoot "runtime"

Write-Host "Downloading Python $PythonVersion..."
Download-File $pythonUrl $runtimeZip
Expand-Archive -Path $runtimeZip -DestinationPath $runtimeRoot

$pthFile = Get-ChildItem $runtimeRoot -Filter "*._pth" | Select-Object -First 1
if (-not $pthFile) {
    throw "Could not find the Python path configuration file."
}
$pthText = Get-Content $pthFile.FullName
$pthText = $pthText -replace "^\s*#\s*import site\s*$", "import site"
Set-Content -Path $pthFile.FullName -Value $pthText

Write-Host "Installing pip and project packages..."
Download-File $getPipUrl $getPip
& (Join-Path $runtimeRoot "python.exe") $getPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    throw "pip installation failed."
}
& (Join-Path $runtimeRoot "python.exe") -m pip install --no-warn-script-location -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Project dependency installation failed."
}

$excluded = @(".git", ".venv", "__pycache__", ".pytest_cache", "dist", "PROJECT_GUIDE.pdf")
Get-ChildItem $projectRoot -Force | Where-Object {
    $excluded -notcontains $_.Name -and $_.Name -ne ".gitignore"
} | ForEach-Object {
    Copy-Item $_.FullName -Destination $stagingRoot -Recurse -Force
}

@'
@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
    echo Portable Python runtime is missing.
    pause
    exit /b 1
)
"%~dp0runtime\python.exe" -m streamlit run "%~dp0app.py"
if errorlevel 1 (
    echo.
    echo Numerica could not start. The error is shown above.
    pause
)
'@ | Set-Content -Path (Join-Path $stagingRoot "Start Numerica.bat") -Encoding ASCII

@'
Numerica Portable

Double-click "Start Numerica.bat" to start the application.
Open http://localhost:8501 if the browser does not open automatically.
'@ | Set-Content -Path (Join-Path $stagingRoot "README.txt") -Encoding UTF8

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $portableZip -CompressionLevel Optimal
Remove-Item $runtimeZip, $getPip -Force

Write-Host ""
Write-Host "Portable package created:"
Write-Host $portableZip

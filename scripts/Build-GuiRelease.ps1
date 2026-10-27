[CmdletBinding()]
param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [string]$DistPath = ".\dist",
    [string]$WorkPath = ".\build\gui-release"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = [IO.Path]::GetFullPath((Join-Path $projectRoot $PythonPath))
$dist = [IO.Path]::GetFullPath((Join-Path $projectRoot $DistPath))
$work = [IO.Path]::GetFullPath((Join-Path $projectRoot $WorkPath))
$spec = Join-Path $projectRoot 'packaging\SearchOrchestrator.spec'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable was not found: $python"
}

& $python -m PyInstaller --noconfirm --clean --distpath $dist --workpath $work $spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$distribution = Join-Path $dist 'SearchOrchestrator'
Copy-Item -LiteralPath (Join-Path $projectRoot 'USER_GUIDE_JA.md') -Destination (Join-Path $distribution 'はじめてガイド.md') -Force
& (Join-Path $PSScriptRoot 'Add-GuiLicenseNotices.ps1') -DistributionPath $distribution
if ($LASTEXITCODE -ne 0) {
    throw "License notice copy failed with exit code $LASTEXITCODE"
}

Write-Host "GUI distribution created: $distribution"
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DistributionPath
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$destination = [System.IO.Path]::GetFullPath($DistributionPath)

if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
    throw "Distribution directory does not exist: $destination"
}

$noticeSource = Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md'
$licenseSource = Join-Path $projectRoot 'LICENSES\\LGPL-3.0.txt'
if (-not (Test-Path -LiteralPath $noticeSource -PathType Leaf) -or
    -not (Test-Path -LiteralPath $licenseSource -PathType Leaf)) {
    throw 'Required license notice source files are missing from the project root.'
}

$licenseDestination = Join-Path $destination 'LICENSES'
New-Item -ItemType Directory -Force -Path $licenseDestination | Out-Null
Copy-Item -LiteralPath $noticeSource -Destination (Join-Path $destination 'THIRD_PARTY_NOTICES.md') -Force
Copy-Item -LiteralPath $licenseSource -Destination (Join-Path $licenseDestination 'LGPL-3.0.txt') -Force

$qtCoreDll = Get-ChildItem -LiteralPath $destination -Recurse -Filter 'Qt6Core.dll' -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $qtCoreDll) {
    Write-Warning 'Qt6Core.dll was not found. Confirm this distribution contains PySide6/Qt shared DLLs before release.'
}

Write-Host "Included third-party notices in: $destination"
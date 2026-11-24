# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup: create a virtual environment, install the dependencies and
# build the models.
# Usage (from this folder, PowerShell):
#   .\setup.ps1
#   .\setup.ps1 -Proxy http://proxy.example.com:912   # behind a corporate proxy
#   .\setup.ps1 -SkipModels                           # install only
#
[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersion = "3.12",
    [string]$Venv = "venv",
    [string]$Proxy = "",
    [switch]$Dev,
    [switch]$SkipInstall,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $Venv)) {
    Write-Host "[setup] creating virtual environment: $Venv"
    if ($Python -eq "py") {
        & py "-$PythonVersion" -m venv $Venv
    } else {
        & $Python -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment (Python 3.11 or newer is required)" }
}

$py = Join-Path $Venv "Scripts\python.exe"
$pipArgs = @()
if ($Proxy) { $pipArgs += @("--proxy", $Proxy) }

if (-not $SkipInstall) {
    $req = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
    Write-Host "[setup] installing $req"
    & $py -m pip install --upgrade pip @pipArgs
    & $py -m pip install -r $req @pipArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("pip install failed. Behind a proxy, pass -Proxy http://<host>:<port>. " +
            "Offline, install from a local wheel folder: " +
            "$py -m pip install --no-index --find-links <wheels_dir> -r $req")
    }
}

Write-Host "[setup] checking OpenVINO devices"
& $py -c "import openvino as ov; c = ov.Core(); [print('  ', d, '-', c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices]"

if (-not $SkipModels) {
    Write-Host "[setup] building the models (about one to two minutes)"
    & $py -m screening_tools setup
    if ($LASTEXITCODE -ne 0) { throw "model setup failed" }
}

Write-Host "[setup] done. Next:"
Write-Host "  .\run.ps1                       # run with the dashboard"
Write-Host "  .\run.ps1 -Headless             # run without the dashboard and print the summary"
Write-Host "  .\run.ps1 -Benchmark            # compare hit scoring on CPU and iGPU"

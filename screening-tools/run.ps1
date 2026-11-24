# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the app from its virtual environment.
# Usage (from this folder, PowerShell):
#   .\run.ps1                                  # dashboard at http://127.0.0.1:8765/
#   .\run.ps1 -Headless -Plates 500
#   .\run.ps1 -Device CPU -Clean
#   .\run.ps1 -Benchmark
#   .\run.ps1 -- run --replay runs\recordings\demo.jsonl   # any CLI arguments after --
#
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Venv = "venv",
    [switch]$Headless,
    [switch]$Benchmark,
    [switch]$Clean,
    [ValidateSet("", "CPU", "GPU")][string]$Device = "",
    [int]$Plates = 0,
    [string]$Config = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) { throw "virtual environment not found; run .\setup.ps1 first" }

if ($Rest) {
    $cliArgs = $Rest
} else {
    $cliArgs = @(if ($Benchmark) { "benchmark" } else { "run" })
    if ($Config) { $cliArgs += @("--config", $Config) }
    if (-not $Benchmark) {
        if ($Headless) { $cliArgs += "--headless" }
        if ($Clean) { $cliArgs += "--clean" }
        if ($Device) { $cliArgs += @("--device", $Device) }
        if ($Plates -gt 0) { $cliArgs += @("--plates", $Plates) }
    }
}
& $py -m screening_tools @cliArgs
exit $LASTEXITCODE

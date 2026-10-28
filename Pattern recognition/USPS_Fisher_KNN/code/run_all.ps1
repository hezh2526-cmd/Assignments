$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
python "$PSScriptRoot\run_experiments.py" --project-root "$ProjectRoot"


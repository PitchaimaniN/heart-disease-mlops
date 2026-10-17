$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    }
    else {
        throw "Python was not found. Install Python and try again."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Creating the virtual environment failed with exit code $LASTEXITCODE."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Upgrading pip failed with exit code $LASTEXITCODE."
}

& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Installing project dependencies failed with exit code $LASTEXITCODE."
}

& $venvPython -m ipykernel install --user --name heart-disease-mlops --display-name "Python (heart-disease-mlops)"
if ($LASTEXITCODE -ne 0) {
    throw "Registering the Jupyter kernel failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Project environment is ready."
Write-Host "Activate it in this terminal with: .\.venv\Scripts\Activate.ps1"
Write-Host "In VS Code, select the 'Python (heart-disease-mlops)' notebook kernel."

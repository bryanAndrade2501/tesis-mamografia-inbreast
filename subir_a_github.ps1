# Ejecutar DESPUÉS de: gh auth login
# Crea el repo en GitHub y sube el código.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$repoName = "tesis-mamografia-inbreast"

gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Primero inicia sesión: gh auth login"
    exit 1
}

gh repo create $repoName `
    --public `
    --source=. `
    --remote=origin `
    --description "Pipeline MATLAB: mejora de mamografías DICOM INbreast con CNN" `
    --push

Write-Host ""
Write-Host "Listo. Clona en otra PC con:"
Write-Host "  git clone https://github.com/TU_USUARIO/$repoName.git"

$packDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$imagesDir = Join-Path $packDir "images"
$pythonExe = "E:\code\conda\python.exe"

if (-not (Test-Path $pythonExe)) {
  $pythonExe = "python"
}

Write-Host "Launching LabelMe on $imagesDir"
Write-Host "Use labels: region_A and region_B"
Start-Process -FilePath $pythonExe -ArgumentList @("-m", "labelme", $imagesDir)

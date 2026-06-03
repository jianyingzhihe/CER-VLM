$packDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $packDir "..\..")
$script = Join-Path $repoRoot "scripts\local\export_paper2_followup_labelme_masks.py"
$pythonExe = "E:\code\conda\python.exe"

if (-not (Test-Path $pythonExe)) {
  $pythonExe = "python"
}

Write-Host "Exporting LabelMe JSON to masks under $packDir\exported_masks"
& $pythonExe $script --pack-dir $packDir

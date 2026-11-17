# Detached, windowless start of run_rev5b.ps1 (it waits for part A itself).
Set-Location $PSScriptRoot
$script = Join-Path $PSScriptRoot "run_rev5b.ps1"
$log = Join-Path $PSScriptRoot "results\log_rev5b_driver.txt"
$p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput $log -PassThru
Write-Output ("launched pid {0}" -f $p.Id)

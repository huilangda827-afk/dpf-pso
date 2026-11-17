# Detached, windowless start of run_rev5a.ps1.  Driver log: results\log_rev5a_driver.txt
Set-Location $PSScriptRoot
$script = Join-Path $PSScriptRoot "run_rev5a.ps1"
$log = Join-Path $PSScriptRoot "results\log_rev5a_driver.txt"
$p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput $log -PassThru
Write-Output ("launched pid {0}" -f $p.Id)

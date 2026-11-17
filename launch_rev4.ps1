# Starts run_rev4.ps1 as a detached, windowless process so that closing the
# IDE terminal cannot kill the experiments (see notes/design_log.md, RD-21
# style launch).  Driver log: results\log_rev4_driver.txt
Set-Location $PSScriptRoot
$script = Join-Path $PSScriptRoot "run_rev4.ps1"
$log = Join-Path $PSScriptRoot "results\log_rev4_driver.txt"
$p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput $log -PassThru
Write-Output ("launched pid {0}" -f $p.Id)

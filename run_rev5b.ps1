# Journal version, second round, part B (design log 18b, 18e, 18f).  Starts
# only after run_rev5a.ps1 has written "all done".  Resumable.  14 workers.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "python"
$W = "14"
$H = @(); foreach ($n in 10, 20) { foreach ($k in 1..40) { $H += ("H{0:D2}_n{1}" -f $k, $n) } }

while (-not (Test-Path results\log_rev5a_driver.txt) -or -not (Select-String -Path results\log_rev5a_driver.txt -Pattern "all done" -Quiet)) {
    Start-Sleep -Seconds 120
}
function Step($name, $argsList) {
    Write-Output ("[{0}] start {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
    & $py src\run_experiments.py @argsList --workers $W --skip-existing *> ("results\log_" + $name + ".txt")
    Write-Output ("[{0}] end {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
}
# 18b. remaining ablation components on the held-out scenarios
Step "heldout_abl2" (@("--tag", "heldout", "--runs", "10", "--instances") + $H + @("--algos", "DPF-PSO_noPG", "DPF-PSO_noA"))
# 18f. spherical encoding for DE and CMA-ES, original and band model
Step "heldout_sph" (@("--tag", "heldout", "--runs", "10", "--instances") + $H + @("--algos", "DE-sph", "CMAES-sph"))
Step "heldout_band_sph" (@("--tag", "heldout_band", "--runs", "10", "--params", '{\"band_along\": true}', "--instances") + $H + @("--algos", "DE-sph", "CMAES-sph"))
# 18e. multi-start local optimizer
Step "heldout_mspowell" (@("--tag", "heldout", "--runs", "10", "--instances") + $H + @("--algos", "MS-Powell"))
Write-Output ("[{0}] start verify" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
& $py src\verify_paths.py heldout heldout_band *> results\log_verify_rev5b.txt
Write-Output ("[{0}] all done" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

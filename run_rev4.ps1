# Journal-version additions (fourth revision): confirmatory test of the
# altitude-band finding on 40 fresh held-out scenarios H41-H80.
# Plan and hypotheses: notes/design_log.md, section 15 (written before any run).
# Resumable: finished (instance, algorithm) files are skipped.
# 16 workers: the machine is also in daily use.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "python"
$W = "16"
$H = @(); foreach ($n in 10, 20) { foreach ($k in 41..80) { $H += ("H{0:D2}_n{1}" -f $k, $n) } }
$A = @("SPSO", "SPSO-T", "DE", "IDP", "DPF-PSO", "DPF-PSO-T")

function Step($name, $argsList) {
    Write-Output ("[{0}] start {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
    & $py src\run_experiments.py @argsList --workers $W --skip-existing *> ("results\log_" + $name + ".txt")
    Write-Output ("[{0}] end {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
}

# 1. band model first (the finding under test)
Step "heldout2_band" (@("--tag", "heldout2_band", "--runs", "10", "--params", '{\"band_along\": true}', "--instances") + $H + @("--algos") + $A)
# 2. original model on the same scenarios
Step "heldout2"      (@("--tag", "heldout2", "--runs", "10", "--instances") + $H + @("--algos") + $A)
# 3. independent safety check of every final path
Write-Output ("[{0}] start verify" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
& $py src\verify_paths.py heldout2 heldout2_band *> results\log_verify_rev4.txt
Write-Output ("[{0}] all done" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

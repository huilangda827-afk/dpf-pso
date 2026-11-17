# Third-revision additions (run after run_all.ps1).  Resumable: finished
# (instance, algorithm) files are skipped.  No method or protocol of the
# second revision is changed; see notes/design_log.md, section 13.
# 12 workers: another job may share the machine.
$ErrorActionPreference = "Continue"
$py = "python"
$W = "12"
$H = @(); foreach ($n in 10, 20) { foreach ($k in 1..40) { $H += ("H{0:D2}_n{1}" -f $k, $n) } }

function Step($name, $argsList) {
    Write-Output ("[{0}] start {1}" -f (Get-Date -Format "HH:mm:ss"), $name)
    & $py src\run_experiments.py @argsList --workers $W --skip-existing *> ("results\log_" + $name + ".txt")
    Write-Output ("[{0}] end {1}" -f (Get-Date -Format "HH:mm:ss"), $name)
}

# 1. CMA-ES receives the same tuning budget as the other tuned methods.  The
#    default configuration was selected, so tuned CMA-ES is identical to
#    CMA-ES; the analysis uses the CMA-ES runs for it (no duplicate runs).
Write-Output ("[{0}] start tune_cmaes" -f (Get-Date -Format "HH:mm:ss"))
if (-not (Test-Path results\tuning\CMAES.json)) { & $py src\tune.py --algos CMAES --workers 20 *> results\log_tune_cmaes.txt }
Write-Output ("[{0}] end tune_cmaes" -f (Get-Date -Format "HH:mm:ss"))
# 2. sensitivity: altitude band enforced along every segment
Step "heldout_band"   (@("--tag","heldout_band","--runs","10","--params",'{\"band_along\": true}',"--instances") + $H + @("--algos","SPSO","SPSO-T","DE","IDP","DPF-PSO","DPF-PSO-T"))
# 3. ablation on the held-out scenarios
Step "heldout_abl"    (@("--tag","heldout","--runs","10","--instances") + $H + @("--algos","DPF-only","DPF-PSO_greedy","DPF-PSO_noG"))
# 4. equal CPU time on the held-out scenarios (strongest baseline of each kind)
Step "heldout_eqtime" (@("--tag","heldout_eqtime","--runs","10","--mult","results\timing\multipliers.json","--instances") + $H + @("--algos","IDP","SPSO","DE"))
Write-Output ("[{0}] all done" -f (Get-Date -Format "HH:mm:ss"))

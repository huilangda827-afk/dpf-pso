# Journal version, second round, part A (design log section 18: 18a, 18c, 18d).
# No method or protocol change.  Resumable.  14 workers: the machine is in daily use.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "python"
$W = "14"
$H = @(); foreach ($n in 10, 20) { foreach ($k in 1..40) { $H += ("H{0:D2}_n{1}" -f $k, $n) } }
$S = @(); foreach ($n in 10, 20) { foreach ($k in 1..6) { $S += ("S{0}_n{1}" -f $k, $n) } }

function Step($name, $argsList) {
    Write-Output ("[{0}] start {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
    & $py src\run_experiments.py @argsList --workers $W --skip-existing *> ("results\log_" + $name + ".txt")
    Write-Output ("[{0}] end {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name)
}

# 18a. equal CPU time for the remaining baselines (budgets multiplied per timing benchmark)
Step "heldout_eqtime_all" (@("--tag", "heldout_eqtime", "--runs", "10", "--mult", "results\timing\multipliers.json", "--instances") + $H + @("--algos", "PSO", "GWO", "WOA", "HHO", "LSHADE"))
# 18d. weight sensitivity b4 = 10 on S1-S6 (mirrors b2 = 10)
Step "b4" (@("--tag", "b4", "--runs", "10", "--params", '{\"b4\": 10}', "--instances") + $S + @("--algos", "PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES", "IDP", "GLS", "DPF-PSO"))
# 18c. plug-in on the held-out scenarios
Step "heldout_plugin" (@("--tag", "heldout", "--runs", "10", "--instances") + $H + @("--algos", "GWO+DPF", "WOA+DPF", "HHO+DPF", "DE+DPF", "LSHADE+DPF"))
Write-Output ("[{0}] start verify" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
& $py src\verify_paths.py heldout heldout_eqtime b4 *> results\log_verify_rev5a.txt
Write-Output ("[{0}] all done" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

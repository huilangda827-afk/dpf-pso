# Second-revision experiment pipeline (run from the project root after tuning).
# Every step can be resumed: finished (instance, algorithm) files are skipped.
$ErrorActionPreference = "Continue"
$py = "python"
$S = @("S1_n10","S2_n10","S3_n10","S4_n10","S5_n10","S6_n10","S1_n20","S2_n20","S3_n20","S4_n20","S5_n20","S6_n20")
$S30 = @("S1_n30","S2_n30","S3_n30","S4_n30","S5_n30","S6_n30")
$H = @(); foreach ($n in 10, 20) { foreach ($k in 1..40) { $H += ("H{0:D2}_n{1}" -f $k, $n) } }
$MAIN = @("PSO","SPSO","GWO","WOA","HHO","DE","LSHADE","CMAES","IDP","GLS","DPF-PSO")
$TUNED = @("DE-T","LSHADE-T","SPSO-T","IDP-T","DPF-PSO-T")
$EXTRA = @("GWO+DPF","WOA+DPF","HHO+DPF","DE+DPF","LSHADE+DPF",
           "DPF-PSO_noG","DPF-PSO_noR","DPF-PSO_noL","DPF-PSO_noA","DPF-PSO_greedy","DPF-only",
           "DE-NP5D","DE-NP10D","SPSO-N100","SPSO-N500")
$BASE = $MAIN | Where-Object { $_ -ne "DPF-PSO" }

function Step($name, $argsList) {
    Write-Output ("[{0}] start {1}" -f (Get-Date -Format "HH:mm:ss"), $name)
    & $py src\run_experiments.py @argsList --skip-existing *> ("results\log_" + $name + ".txt")
    Write-Output ("[{0}] end {1}" -f (Get-Date -Format "HH:mm:ss"), $name)
}

Write-Output ("[{0}] start timing" -f (Get-Date -Format "HH:mm:ss"))
& $py src\timing_benchmark.py --tag timing --runs 5 *> results\log_timing.txt
Write-Output ("[{0}] end timing" -f (Get-Date -Format "HH:mm:ss"))

Step "main"    (@("--tag","main","--runs","30","--instances") + $S + @("--algos") + $MAIN + $TUNED + $EXTRA)
Step "heldout" (@("--tag","heldout","--runs","10","--instances") + $H + @("--algos") + $MAIN + $TUNED)
Step "public"  (@("--tag","public","--runs","30","--instances","C1_n10","C1_n20","--algos") + $MAIN + $TUNED)
Step "eqtime"  (@("--tag","eqtime","--runs","30","--mult","results\timing\multipliers.json","--instances") + $S + @("--algos") + $BASE)
Step "n30"     (@("--tag","n30","--runs","20","--instances") + $S30 + @("--algos") + $MAIN)
Step "budget5" (@("--tag","budget5","--runs","10","--budget-mult","5","--instances") + $S + @("--algos") + $MAIN)
Step "b2"      (@("--tag","b2","--runs","10","--params",'{\"b2\": 10.0}',"--instances") + $S + @("--algos") + $MAIN)
Write-Output ("[{0}] all done" -f (Get-Date -Format "HH:mm:ss"))

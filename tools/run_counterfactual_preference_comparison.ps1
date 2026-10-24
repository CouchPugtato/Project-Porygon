[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot '../config/experimental/counterfactual_preference_comparison.json'),
    [switch]$SummaryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent

function Resolve-ProjectPath([string]$Value) {
    return [IO.Path]::GetFullPath($Value, $repoRoot)
}

function Get-Metric($Report, [string]$Split, [string]$Name, [string]$When = 'after') {
    $value = $Report.$When.$Split.$Name
    if ($null -eq $value -or [double]::IsNaN([double]$value) -or
            [double]::IsInfinity([double]$value)) {
        throw "Missing or non-finite metric: $When.$Split.$Name"
    }
    return [double]$value
}

function Read-FitReport([string]$Path, $Trial, [string]$Objective) {
    $report = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    $minimumVersion = if ($Objective -eq 'preference') { 12 } else { 11 }
    $objectiveName = if ($Objective -eq 'preference') {
        'rollout_confidence_weighted_pairwise_preference'
    } else {
        'rollout_confidence_weighted_pair_gap_huber_regression'
    }
    if ($report.metrics_version -lt $minimumVersion -or
            $report.training_objective -ne $objectiveName -or
            $report.diagnostic -ne 'paired_counterfactual_q_fit' -or
            $report.holdout_role -ne 'development' -or
            $report.holdout_source -ne 'stable_pair_split' -or
            $report.representation -ne 'current_hidden_and_selected_action_semantics' -or
            $report.head_mode -ne 'selected' -or
            -not $report.assessment.training_completed) {
        throw "Report is not a completed $Objective development fit: $Path"
    }
    if ((Resolve-ProjectPath $report.counterfactual_batch) -ne $batchPath -or
            (Resolve-ProjectPath $report.encoder_checkpoint) -ne $checkpointPath) {
        throw "Report uses a different batch or checkpoint: $Path"
    }
    $expected = @{
        validation_seed = $Trial.validation_seed
        shuffle_seed = $config.shuffle_seed
        epochs_requested = $config.epochs
        minibatch_pairs = $config.minibatch_pairs
        latent_dim = $config.latent_dim
        early_stop_patience = $config.early_stop_patience
        learning_rate = $config.learning_rate
        l2_coefficient = $config.l2_coefficient
        minimum_pair_confidence = $config.minimum_pair_confidence
    }
    foreach ($name in $expected.Keys) {
        $tolerance = [math]::Max(1e-12, [math]::Abs([double]$expected[$name]) * 1e-6)
        $actual = [double]$report.$name
        if ([double]::IsNaN($actual) -or [double]::IsInfinity($actual) -or
                [math]::Abs($actual - [double]$expected[$name]) -gt $tolerance) {
            throw "Report setting '$name' differs from the comparison config: $Path"
        }
    }
    foreach ($split in @('train', 'selection', 'holdout')) {
        if ((Get-Metric $report $split 'nonfinite_values') -ne 0 -or
                (Get-Metric $report $split 'discordant_pairs') -le 0) {
            throw "Report has non-finite outputs or no usable $split pairs: $Path"
        }
    }
    return $report
}

function Get-ComparisonRow($Trial, $Gap, $Preference) {
    $row = [ordered]@{ validation_seed = $Trial.validation_seed; report = $Trial.report_name }
    foreach ($split in @('selection', 'holdout')) {
        foreach ($entry in @(@('gap', $Gap), @('preference', $Preference))) {
            $name = $entry[0]
            $report = $entry[1]
            $before = Get-Metric $report $split 'confidence_weighted_pairwise_preference_loss' 'before'
            if ($before -le 0) { throw "Cannot compare a zero initial preference loss" }
            $row["${name}_${split}_preference_loss_ratio"] =
                (Get-Metric $report $split 'confidence_weighted_pairwise_preference_loss') / $before
            $row["${name}_${split}_ranking"] =
                Get-Metric $report $split 'confidence_weighted_pair_ranking_accuracy'
        }
    }
    foreach ($split in @('train', 'selection', 'holdout')) {
        if ((Get-Metric $Gap $split 'pairs') -ne (Get-Metric $Preference $split 'pairs')) {
            throw "The two objectives used different $split pair counts"
        }
    }
    $beforeGap = Get-Metric $Preference 'holdout' 'confidence_weighted_pair_gap_huber_loss' 'before'
    $row.preference_holdout_gap_loss_ratio = if ($beforeGap -gt 0) {
        (Get-Metric $Preference 'holdout' 'confidence_weighted_pair_gap_huber_loss') / $beforeGap
    } else { $null }
    $row.holdout_pairs = Get-Metric $Preference 'holdout' 'discordant_pairs'
    $row.best_epoch = $Preference.best_epoch
    $row.holdout_mean_absolute_advantage = Get-Metric $Preference 'holdout' 'mean_absolute_advantage'
    $row.holdout_max_absolute_advantage = Get-Metric $Preference 'holdout' 'max_absolute_advantage'
    $row.action_signal = $Preference.assessment.action_signal_detected
    return [pscustomobject]$row
}

$config = Get-Content -Raw -LiteralPath (Resolve-ProjectPath $ConfigPath) | ConvertFrom-Json
if (@($config.trials).Count -eq 0 -or $config.omp_threads -lt 1 -or $config.omp_threads -gt 4) {
    throw 'The comparison needs trials and an OMP thread count between 1 and 4'
}
$batchPath = Resolve-ProjectPath $config.batch
$checkpointPath = Resolve-ProjectPath $config.checkpoint
$reportDirectory = Resolve-ProjectPath $config.report_directory
$trainerPath = Join-Path $repoRoot 'build-fresh/showdown_client.exe'
$gapReports = @()

# Validate existing evidence before launching any training.
foreach ($trial in $config.trials) {
    $gapReports += Read-FitReport (Resolve-ProjectPath $trial.gap_report) $trial 'gap'
    $reportPath = Join-Path $reportDirectory $trial.report_name
    if (Test-Path -LiteralPath $reportPath) {
        $null = Read-FitReport $reportPath $trial 'preference'
    } elseif ($SummaryOnly) {
        throw "Preference report is missing: $reportPath"
    }
}

$previousOmpThreads = $env:PORYGON_OMP_THREADS
Push-Location $repoRoot
try {
    if (-not $SummaryOnly) {
        if (-not (Test-Path -LiteralPath $trainerPath)) { throw "Trainer is missing: $trainerPath" }
        New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
    }
    $env:PORYGON_OMP_THREADS = [string]$config.omp_threads
    $rows = @()
    for ($index = 0; $index -lt $config.trials.Count; ++$index) {
        $trial = $config.trials[$index]
        $reportPath = Join-Path $reportDirectory $trial.report_name
        if (-not (Test-Path -LiteralPath $reportPath)) {
            $arguments = @(
                '--check-counterfactual-q-fit', $batchPath, $checkpointPath, $reportPath,
                '--counterfactual-q-representation', 'semantic',
                '--counterfactual-q-head', 'selected',
                '--counterfactual-q-objective', 'preference',
                '--counterfactual-final-confirmation', '0',
                '--counterfactual-min-confidence', [string]$config.minimum_pair_confidence,
                '--epochs', [string]$config.epochs,
                '--learning-rate', [string]$config.learning_rate,
                '--validation-seed', [string]$trial.validation_seed,
                '--seed', [string]$config.shuffle_seed,
                '--action-q-minibatch-pairs', [string]$config.minibatch_pairs,
                '--action-q-latent-dim', [string]$config.latent_dim,
                '--action-q-early-stop-patience', [string]$config.early_stop_patience,
                '--action-q-l2', [string]$config.l2_coefficient
            )
            Write-Host "[preference-comparison] split=$($index + 1)/$($config.trials.Count) seed=$($trial.validation_seed)"
            & $trainerPath @arguments
            if ($LASTEXITCODE -ne 0) { throw "Preference fit failed with exit code $LASTEXITCODE" }
        } else {
            Write-Host "[preference-comparison] reusing $($trial.report_name)"
        }
        $preference = Read-FitReport $reportPath $trial 'preference'
        $rows += Get-ComparisonRow $trial $gapReports[$index] $preference
    }
    $summary = [ordered]@{
        diagnostic = 'counterfactual_objective_comparison'
        direction_step = 'project_direction.txt: sequence of work, step 1'
        assessment = 'development_only'
        independent_replications = $false
        fresh_confirmation_required = $true
        batch = $batchPath
        checkpoint = $checkpointPath
        configuration = $config
        trials = $rows
    }
    $summaryPath = Join-Path $reportDirectory $config.summary_name
    $temporaryPath = "$summaryPath.tmp"
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporaryPath -Encoding utf8
    Move-Item -LiteralPath $temporaryPath -Destination $summaryPath -Force
    $rows | Select-Object validation_seed, best_epoch, holdout_pairs,
        @{Name='GapRank'; Expression={[math]::Round($_.gap_holdout_ranking, 3)}},
        @{Name='PreferenceRank'; Expression={[math]::Round($_.preference_holdout_ranking, 3)}},
        @{Name='GapPrefLossRatio'; Expression={[math]::Round($_.gap_holdout_preference_loss_ratio, 3)}},
        @{Name='PreferenceLossRatio'; Expression={[math]::Round($_.preference_holdout_preference_loss_ratio, 3)}},
        action_signal | Format-Table -AutoSize | Out-Host
    Write-Host "[preference-comparison] summary=$summaryPath"
    Write-Host '[preference-comparison] Development splits reuse one collection; fresh confirmation is still required.'
} finally {
    $env:PORYGON_OMP_THREADS = $previousOmpThreads
    Pop-Location
}

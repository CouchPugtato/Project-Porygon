Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_counterfactual_preference_comparison.ps1'
$repoRoot = Split-Path $PSScriptRoot -Parent
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar)
$fixtureDirectory = Join-Path $temporaryRoot ('porygon-preference-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $fixtureDirectory | Out-Null
$originalOmpThreads = $env:PORYGON_OMP_THREADS

function Write-Fixture([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Assert-Rejected([scriptblock]$Action, [string]$ExpectedMessage) {
    $caught = $null
    try { & $Action } catch { $caught = $_.Exception.Message }
    if ($null -eq $caught -or -not $caught.Contains($ExpectedMessage)) {
        throw "Expected rejection containing '$ExpectedMessage'; received '$caught'"
    }
}

try {
    $config = Get-Content -Raw (Join-Path $repoRoot 'config/experimental/counterfactual_preference_comparison.json') |
        ConvertFrom-Json
    $config.report_directory = $fixtureDirectory
    $config.summary_name = 'summary.json'
    $config.trials = @($config.trials[0])
    $trial = $config.trials[0]
    $trial.gap_report = Join-Path $fixtureDirectory 'gap.json'
    $preferencePath = Join-Path $fixtureDirectory $trial.report_name
    $configPath = Join-Path $fixtureDirectory 'config.json'
    Write-Fixture $configPath $config

    $metrics = [ordered]@{
        pairs = 60; discordant_pairs = 53; nonfinite_values = 0
        confidence_weighted_pairwise_preference_loss = 0.65
        confidence_weighted_pair_ranking_accuracy = 0.60
        confidence_weighted_pair_gap_huber_loss = 0.35
        mean_absolute_advantage = 0.1; max_absolute_advantage = 0.5
    }
    $gap = [ordered]@{
        metrics_version = 11
        diagnostic = 'paired_counterfactual_q_fit'
        training_objective = 'rollout_confidence_weighted_pair_gap_huber_regression'
        holdout_role = 'development'; holdout_source = 'stable_pair_split'
        representation = 'current_hidden_and_selected_action_semantics'; head_mode = 'selected'
        counterfactual_batch = $config.batch; encoder_checkpoint = $config.checkpoint
        validation_seed = $trial.validation_seed; shuffle_seed = $config.shuffle_seed
        epochs_requested = $config.epochs; minibatch_pairs = $config.minibatch_pairs
        latent_dim = $config.latent_dim; early_stop_patience = $config.early_stop_patience
        learning_rate = $config.learning_rate; l2_coefficient = $config.l2_coefficient
        minimum_pair_confidence = $config.minimum_pair_confidence; best_epoch = 20
        assessment = @{training_completed = $true; action_signal_detected = $false}
        before = @{train = $metrics; selection = $metrics; holdout = $metrics}
        after = @{train = $metrics; selection = $metrics; holdout = $metrics}
    }
    Write-Fixture $trial.gap_report $gap
    $preference = $gap | ConvertTo-Json -Depth 12 | ConvertFrom-Json
    $preference.metrics_version = 12
    $preference.training_objective = 'rollout_confidence_weighted_pairwise_preference'
    $preference.after.holdout.confidence_weighted_pairwise_preference_loss = 0.585
    Write-Fixture $preferencePath $preference

    $env:PORYGON_OMP_THREADS = '2'
    & $runner -ConfigPath $configPath -SummaryOnly
    $summary = Get-Content -Raw (Join-Path $fixtureDirectory 'summary.json') | ConvertFrom-Json
    if ([math]::Abs($summary.trials[0].preference_holdout_preference_loss_ratio - 0.9) -gt 1e-9 -or
            $summary.independent_replications -or -not $summary.fresh_confirmation_required -or
            $env:PORYGON_OMP_THREADS -ne '2') {
        throw 'Completed reports must resume, retain correct loss ratios, and restore thread settings'
    }

    $preference.learning_rate = 0.002
    Write-Fixture $preferencePath $preference
    Assert-Rejected { & $runner -ConfigPath $configPath -SummaryOnly } "setting 'learning_rate'"
    $preference.learning_rate = $config.learning_rate
    $preference.after.holdout.pairs = 61
    Write-Fixture $preferencePath $preference
    Assert-Rejected { & $runner -ConfigPath $configPath -SummaryOnly } 'different holdout pair counts'
    $preference.after.holdout.pairs = 60
    $preference.after.holdout.nonfinite_values = 1
    Write-Fixture $preferencePath $preference
    Assert-Rejected { & $runner -ConfigPath $configPath -SummaryOnly } 'non-finite outputs'

    Remove-Item -LiteralPath $preferencePath
    Assert-Rejected { & $runner -ConfigPath $configPath -SummaryOnly } 'Preference report is missing'
    Write-Host 'preference comparison tests passed'
} finally {
    $env:PORYGON_OMP_THREADS = $originalOmpThreads
    # Every fixture is created under this unique temporary directory.
    if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($fixtureDirectory)) -ne $temporaryRoot) {
        throw "Refusing to remove fixtures outside the temporary root: $fixtureDirectory"
    }
    Remove-Item -LiteralPath $fixtureDirectory -Recurse -Force
}

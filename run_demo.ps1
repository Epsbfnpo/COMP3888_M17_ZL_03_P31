param(
    [string]$Cohort = "",
    [string]$DataRoot = "",
    [int]$Patients = 2,
    [string]$PatientIds = "",
    [ValidateSet("full", "dashboard")]
    [string]$Mode = "full",
    [switch]$SkipEvaluation,
    [switch]$NoDashboard,
    [int]$ViewerRenderMaxSide = 560,
    [int]$ViewerPrefetchRadius = 0,
    [switch]$ViewerTiming
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
}

function Assert-AsciiPath([string]$PathText, [string]$Name) {
    if ($PathText -match '[^\x00-\x7F]') {
        Write-Host "ERROR: $Name contains non-ASCII characters:" -ForegroundColor Red
        Write-Host "  $PathText" -ForegroundColor Red
        Write-Host "ITK/Elastix on Windows can fail on paths containing Chinese/non-ASCII characters." -ForegroundColor Yellow
        Write-Host "Move the repository/data to an English-only path and run again." -ForegroundColor Yellow
        exit 2
    }
}

function Invoke-PythonStep([string]$Title, [string[]]$CommandArgs, [switch]$AllowFailure) {
    Write-Section $Title
    Write-Host ("$Python " + ($CommandArgs -join " ")) -ForegroundColor DarkGray
    & $Python @CommandArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        if ($AllowFailure) {
            Write-Host "WARNING: step returned exit code $code. Continuing so the dashboard can still be opened." -ForegroundColor Yellow
            return $false
        }
        Write-Host "ERROR: step failed with exit code $code." -ForegroundColor Red
        exit $code
    }
    return $true
}

Write-Section "P31 Longitudinal Lesion Tracking Demo Launcher"
Write-Host "Repository: $RepoRoot"
Assert-AsciiPath $RepoRoot "Repository path"

# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "No .venv was found at $Python" -ForegroundColor Yellow
    $answer = Read-Host "Create Python 3.11 .venv and install requirements now? [Y/n]"
    if ([string]::IsNullOrWhiteSpace($answer) -or $answer.Trim().ToLower() -eq "y") {
        & py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        & $Python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    else {
        Write-Host "Create/activate the project .venv first, then run this launcher again." -ForegroundColor Yellow
        exit 2
    }
}
Write-Host "Python: $Python" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Cohort selection
# -----------------------------------------------------------------------------
$Cohort = $Cohort.Trim().ToUpper()
$DefaultARoot = Join-Path $RepoRoot "data\cohort_a"
$DefaultBRoot = Join-Path $RepoRoot "data\cohort_b"

if ([string]::IsNullOrWhiteSpace($Cohort)) {
    $aExists = Test-Path $DefaultARoot
    $bExists = Test-Path $DefaultBRoot
    if ($aExists -and -not $bExists) {
        $Cohort = "A"
        Write-Host "Detected Cohort A data folder." -ForegroundColor Green
    }
    elseif ($bExists -and -not $aExists) {
        $Cohort = "B"
        Write-Host "Detected Cohort B data folder." -ForegroundColor Green
    }
    else {
        $choice = Read-Host "Choose dataset: Cohort A or Cohort B? [A/B, default A]"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "A" }
        $Cohort = $choice.Trim().ToUpper()
    }
}
if ($Cohort -notin @("A", "B")) {
    Write-Host "ERROR: -Cohort must be A or B." -ForegroundColor Red
    exit 2
}

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $defaultRoot = if ($Cohort -eq "A") { $DefaultARoot } else { $DefaultBRoot }
    if (Test-Path $defaultRoot) {
        $DataRoot = (Resolve-Path $defaultRoot).Path
    }
    else {
        $DataRoot = Read-Host "Enter the full Cohort $Cohort data root"
    }
}
if (-not (Test-Path $DataRoot)) {
    Write-Host "ERROR: Data root does not exist: $DataRoot" -ForegroundColor Red
    exit 2
}
$DataRoot = (Resolve-Path $DataRoot).Path
Assert-AsciiPath $DataRoot "Data root"

if ($Patients -lt 1 -and [string]::IsNullOrWhiteSpace($PatientIds)) {
    $Patients = 2
}

$cohortSlug = if ($Cohort -eq "A") { "a" } else { "b" }
$selectionLabel = if ([string]::IsNullOrWhiteSpace($PatientIds)) { "$Patients" } else { "selected" }
$DemoTag = "demo_${cohortSlug}${selectionLabel}"
$ManifestOut = Join-Path $RepoRoot "outputs\$DemoTag\manifest"
$PipelineOut = Join-Path $RepoRoot "outputs\$DemoTag\tracking"
$RegistrationRoot = Join-Path $RepoRoot "outputs\registration"

$PairFileName = if ($Cohort -eq "A") { "cohort_a_subset_pairs.csv" } else { "cohort_b_subset_pairs.csv" }
$PairsPath = Join-Path $ManifestOut $PairFileName
$SelectedPairsPath = Join-Path $PipelineOut "selected_pairs_11.csv"  # legacy filename used by the pipeline
$FeaturesPath = Join-Path $PipelineOut "aligned_lesion_features.csv"
$MatchesPath = Join-Path $PipelineOut "lesion_matches.csv"
$PairCostsPath = Join-Path $PipelineOut "lesion_pair_costs.csv"
$MatrixDir = Join-Path $PipelineOut "lesion_pair_cost_matrices"
$EvalSummaryPath = Join-Path $PipelineOut "lesion_tracking_summary.csv"
$EvalDetailsPath = Join-Path $PipelineOut "lesion_tracking_details.csv"
$TrackingSummaryPath = Join-Path $PipelineOut "tracking_patient_summary.csv"

Write-Host ""
Write-Host "Cohort      : $Cohort" -ForegroundColor Green
Write-Host "Data root   : $DataRoot"
Write-Host "Demo output : $PipelineOut"
Write-Host "Mode        : $Mode"
Write-Host "Viewer max  : ${ViewerRenderMaxSide}px"

if ($Cohort -eq "A") {
    $env:COHORT_A_ROOT = $DataRoot
    $env:COHORT_B_ROOT = ""
    $env:P31_DATASET_LABEL = "Cohort A"
}
else {
    $env:COHORT_B_ROOT = $DataRoot
    $env:COHORT_A_ROOT = ""
    $env:P31_DATASET_LABEL = "Cohort B"
}
$env:DATA_ROOT = $DataRoot
if ($ViewerRenderMaxSide -lt 256) { $ViewerRenderMaxSide = 256 }
if ($ViewerRenderMaxSide -gt 1200) { $ViewerRenderMaxSide = 1200 }
if ($ViewerPrefetchRadius -lt 0) { $ViewerPrefetchRadius = 0 }
if ($ViewerPrefetchRadius -gt 24) { $ViewerPrefetchRadius = 24 }
$env:P31_VIEWER_MAX_SIDE = "$ViewerRenderMaxSide"
$env:P31_VIEWER_PREFETCH_RADIUS = "$ViewerPrefetchRadius"
$env:P31_VIEWER_TIMING = if ($ViewerTiming) { "1" } else { "0" }

# -----------------------------------------------------------------------------
# Full preprocessing/tracking pipeline
# -----------------------------------------------------------------------------
if ($Mode -eq "full") {
    New-Item -ItemType Directory -Force $ManifestOut | Out-Null
    New-Item -ItemType Directory -Force $PipelineOut | Out-Null

    $prepareTool = if ($Cohort -eq "A") {
        "tools/prepare_cohort_a_subset.py"
    }
    else {
        "tools/prepare_cohort_b_subset.py"
    }

    $prepareArgs = @(
        $prepareTool,
        "--root", $DataRoot,
        "--out-dir", $ManifestOut,
        "--path-mode", "relative-to-root"
    )
    if (-not [string]::IsNullOrWhiteSpace($PatientIds)) {
        $prepareArgs += @("--patient-ids", $PatientIds)
    }
    else {
        $prepareArgs += @("--max-patients", "$Patients")
    }
    Invoke-PythonStep "STEP 1 - Prepare Cohort $Cohort manifest" $prepareArgs | Out-Null

    if (-not (Test-Path $PairsPath)) {
        Write-Host "ERROR: pair manifest was not generated: $PairsPath" -ForegroundColor Red
        exit 2
    }
    $pairRows = @(Import-Csv $PairsPath)
    $ActualPatients = $pairRows.Count
    if ($ActualPatients -lt 1) {
        Write-Host "ERROR: the generated pair manifest contains no patients." -ForegroundColor Red
        exit 2
    }
    Write-Host "Selected $ActualPatients patient(s):" -ForegroundColor Green
    $pairRows | ForEach-Object { Write-Host "  - $($_.patient_id)" }

    $pipelineArgs = @(
        "tools/run_tracking_pipeline_11.py",
        "--pairs", $PairsPath,
        "--data-root", $DataRoot,
        "--out-dir", $PipelineOut,
        "--expected-patients", "$ActualPatients",
        "--pet-feature", "none",
        "--pet-weight", "0",
        "--max-bl-per-fu", "3"
    )
    Invoke-PythonStep "STEP 2 - Registration -> features -> costs -> matcher" $pipelineArgs | Out-Null

    # Cohort A has expert-reference data. Evaluation is useful for the demo, but
    # it is deliberately non-fatal so a missing/failed GT case does not prevent
    # the visual dashboard from opening.
    if ($Cohort -eq "A" -and -not $SkipEvaluation) {
        $evaluationArgs = @(
            "tools/run_tracking_evaluation_batch.py",
            "--pairs", $SelectedPairsPath,
            "--features", $FeaturesPath,
            "--data-root", $DataRoot,
            "--out-dir", $PipelineOut,
            "--matrix-dir", $MatrixDir,
            "--expected-patients", "0",
            "--pet-feature", "none",
            "--pet-weight", "0",
            "--max-bl-per-fu", "3",
            "--max-map-distance-vox", "30",
            "--reuse-matrices"
        )
        Invoke-PythonStep "STEP 3 - Cohort A expert ground-truth evaluation" $evaluationArgs -AllowFailure | Out-Null
    }
    elseif ($Cohort -eq "B") {
        Write-Section "STEP 3 - Ground-truth evaluation"
        Write-Host "Skipped: Cohort B does not use the Cohort A expert correspondence evaluation." -ForegroundColor Yellow
    }
    else {
        Write-Section "STEP 3 - Ground-truth evaluation"
        Write-Host "Skipped by request." -ForegroundColor Yellow
    }

    if (-not (Test-Path $MatchesPath)) {
        Write-Host "ERROR: matcher output does not exist: $MatchesPath" -ForegroundColor Red
        exit 2
    }

    $summaryArgs = @(
        "tools/summarise_tracking_results.py",
        "--matches", $MatchesPath,
        "--out", $TrackingSummaryPath
    )
    if (Test-Path $EvalSummaryPath) {
        $summaryArgs += @("--evaluation-summary", $EvalSummaryPath)
    }
    Invoke-PythonStep "STEP 4 - Build per-patient tracking summary" $summaryArgs | Out-Null
}
else {
    Write-Section "Dashboard-only mode"
    if (-not (Test-Path $SelectedPairsPath)) {
        Write-Host "ERROR: existing demo outputs were not found:" -ForegroundColor Red
        Write-Host "  $SelectedPairsPath" -ForegroundColor Red
        Write-Host "Run once with -Mode full first." -ForegroundColor Yellow
        exit 2
    }
}

# -----------------------------------------------------------------------------
# Auto-fill Streamlit via environment variables
# -----------------------------------------------------------------------------
$env:P31_PAIR_MANIFEST = $SelectedPairsPath
$env:P31_REGISTRATION_ROOT = $RegistrationRoot
$env:P31_MATCH_RESULTS = $MatchesPath
$env:P31_ALIGNED_FEATURES = $FeaturesPath
$env:P31_COST_MATRIX_DIR = $MatrixDir
$env:P31_PAIR_COSTS = $PairCostsPath
$env:P31_EVAL_DETAILS = if (Test-Path $EvalDetailsPath) { $EvalDetailsPath } else { "" }
$env:P31_EVAL_SUMMARY = if (Test-Path $EvalSummaryPath) { $EvalSummaryPath } else { "" }

Write-Section "Demo paths"
Write-Host "Pair manifest                 : $env:P31_PAIR_MANIFEST"
Write-Host "Data root                     : $env:DATA_ROOT"
Write-Host "Registration output root      : $env:P31_REGISTRATION_ROOT"
Write-Host "Matching results CSV          : $env:P31_MATCH_RESULTS"
Write-Host "Aligned lesion features CSV   : $env:P31_ALIGNED_FEATURES"
Write-Host "Cost matrix directory         : $env:P31_COST_MATRIX_DIR"
Write-Host "Pair-cost details CSV         : $env:P31_PAIR_COSTS"
Write-Host "GT evaluation details CSV     : $env:P31_EVAL_DETAILS"
Write-Host "GT evaluation summary CSV     : $env:P31_EVAL_SUMMARY"
Write-Host "Viewer render max side         : $env:P31_VIEWER_MAX_SIDE px"
Write-Host "Viewer prefetch radius         : +/-$env:P31_VIEWER_PREFETCH_RADIUS slices"
Write-Host "Viewer timing logs             : $env:P31_VIEWER_TIMING"
Write-Host "Tracking patient summary CSV  : $TrackingSummaryPath"

if ($NoDashboard) {
    Write-Host ""
    Write-Host "Pipeline complete. Dashboard launch skipped (-NoDashboard)." -ForegroundColor Green
    exit 0
}

Write-Section "STEP 5 - Launch Streamlit dashboard"
Write-Host "The sidebar fields are auto-filled from this launcher's environment variables." -ForegroundColor Green
Write-Host "Press Ctrl+C when you want to stop the dashboard; generated files remain on disk." -ForegroundColor Yellow
& $Python -m streamlit run tools/align_longitudinal_patient_v4_mapped.py
exit $LASTEXITCODE

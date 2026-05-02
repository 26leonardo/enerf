#!/usr/bin/env powershell
# E-NeRF Quick Start Script per Windows
# Uso: .\quickstart.ps1 -setup  (primo run)
#      .\quickstart.ps1 -train   (start training)
#      .\quickstart.ps1 -status  (monitora)

param(
    [ValidateSet("setup", "train", "inference", "gui", "status", "download")]
    [string]$Action = "status"
)

# Colori per output
$Colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
}

function Write-Colored {
    param($Message, $Color)
    Write-Host $Message -ForegroundColor $Color
}

function Get-RepoRoot {
    # Trova cartella enerf (dove è questo script)
    if (Test-Path ".\main_nerf.py") {
        return (Get-Location).Path
    } else {
        Write-Colored "❌ Script deve essere eseguito dalla cartella enerf" $Colors.Error
        exit 1
    }
}

$REPO_ROOT = Get-RepoRoot
$CONDA_ENV = "enerf"
$DATA_DIR = "$REPO_ROOT\data"
$EXP_DIR = "$REPO_ROOT\exp"
$TENSORBOARD_PORT = 6006

Write-Colored "
╔════════════════════════════════════════╗
║   E-NeRF Quick Start (RTX 4060 Ti)     ║
║   Repo root: $REPO_ROOT
║   Conda env: $CONDA_ENV
║   Data dir: $DATA_DIR
╚════════════════════════════════════════╝
" $Colors.Info

# ─────────────────────────────────────────────────────────────
# SETUP: Configura environment Conda
# ─────────────────────────────────────────────────────────────

function Invoke-Setup {
    Write-Colored "
    🔧 SETUP ENVIRONMENT
    ─────────────────────" $Colors.Info

    # Verifica conda
    Write-Colored "  [1/8] Verificando conda..." $Colors.Info
    $conda = conda --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "  ❌ Miniconda non trovato!" $Colors.Error
        Write-Colored "     Installare da: https://docs.conda.io/projects/miniconda" $Colors.Warning
        Write-Colored "     URL: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html" $Colors.Warning
        exit 1
    }
    Write-Colored "  ✓ Conda trovato: $conda" $Colors.Success

    # Verifica NVIDIA drivers
    Write-Colored "  [2/8] Verificando NVIDIA drivers..." $Colors.Info
    $nvidia = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "  ❌ NVIDIA drivers non trovati!" $Colors.Error
        Write-Colored "     Installare da: https://www.nvidia.com/Download/driverDetails.aspx" $Colors.Warning
        exit 1
    }
    Write-Colored "  ✓ GPU trovata: $nvidia" $Colors.Success

    # Verifica Visual Studio Build Tools
    Write-Colored "  [3/8] Verificando Visual Studio Build Tools..." $Colors.Info
    $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    $clExe = "$vsPath\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
    if (-not (Test-Path $clExe)) {
        Write-Colored "  ⚠️  Visual Studio Build Tools non trovato" $Colors.Warning
        Write-Colored "     CUDA compilation potrebbe fallire!" $Colors.Warning
        Write-Colored "     Installare da: https://visualstudio.microsoft.com/downloads/" $Colors.Warning
        Write-Colored "     Seleziona: Visual Studio Build Tools 2023 + Desktop C++ development" $Colors.Warning
        Write-Colored "     Continuo comunque..." $Colors.Warning
    } else {
        Write-Colored "  ✓ Visual Studio Build Tools trovato" $Colors.Success
    }

    # Verifica CUDA
    Write-Colored "  [4/8] Verificando CUDA Toolkit..." $Colors.Info
    $nvcc = nvcc --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "  ⚠️  NVCC non trovato (CUDA Toolkit non installato?)" $Colors.Warning
        Write-Colored "     Installare CUDA 12.1 da: https://developer.nvidia.com/cuda-downloads" $Colors.Warning
        Write-Colored "     Continuo comunque..." $Colors.Warning
    } else {
        Write-Colored "  ✓ NVCC trovato: $(($nvcc -split '\n')[0])" $Colors.Success
    }

    # Crea cartelle
    Write-Colored "  [5/8] Creando cartelle..." $Colors.Info
    @($DATA_DIR, $EXP_DIR, "$REPO_ROOT\logs", "$REPO_ROOT\configs_custom") | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
            Write-Colored "       ✓ Creata: $_" $Colors.Success
        }
    }

    # Verifica/crea Conda env
    Write-Colored "  [6/8] Setup Conda environment..." $Colors.Info
    $envExists = conda env list | Select-String $CONDA_ENV
    
    if ($envExists) {
        Write-Colored "  ✓ Environment '$CONDA_ENV' esiste già" $Colors.Success
    } else {
        Write-Colored "     Creando environment (questo richiede ~5 min)..." $Colors.Warning
        conda env create -f environment.yml
        if ($LASTEXITCODE -ne 0) {
            Write-Colored "  ❌ Errore creazione environment!" $Colors.Error
            Write-Colored "     Prova: conda env create -f environment.yml --force-reinstall" $Colors.Info
            exit 1
        }
        Write-Colored "  ✓ Environment creato" $Colors.Success
    }

    # Testa import moduli CUDA
    Write-Colored "  [7/8] Compilando moduli CUDA (primo run, ~2 min)..." $Colors.Info
    
    conda run -n $CONDA_ENV python -c "
try:
    from gridencoder import GridEncoder
    print('  ✓ gridencoder OK')
except Exception as e:
    print(f'  ❌ gridencoder: {e}')
    exit(1)

try:
    from raymarching import raymarching
    print('  ✓ raymarching OK')
except Exception as e:
    print(f'  ❌ raymarching: {e}')
    exit(1)

try:
    from shencoder import SHEncoder
    print('  ✓ shencoder OK')
except Exception as e:
    print(f'  ❌ shencoder: {e}')
    exit(1)
"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "  ❌ Errore compilazione moduli!" $Colors.Error
        Write-Colored "     Vedi troubleshooting sezione 7.1/7.2" $Colors.Warning
        exit 1
    }
    Write-Colored "  ✓ Moduli compilati" $Colors.Success

    # Verifica config
    Write-Colored "  [8/8] Verificando config file..." $Colors.Info
    if (Test-Path "$REPO_ROOT\configs_custom\rtx4060ti_optimal.txt") {
        Write-Colored "  ✓ Config trovato: rtx4060ti_optimal.txt" $Colors.Success
    } else {
        Write-Colored "  ⚠ Config non trovato, creando default..." $Colors.Warning
        # Copia default config
        Copy-Item "$REPO_ROOT\configs\mocapDesk2_enerf.txt" "$REPO_ROOT\configs_custom\rtx4060ti_optimal.txt" -Force 2>$null
    }

    Write-Colored "
    ✅ Setup completato!
    
    Prossimi step:
    1) Scarica dataset: .\quickstart.ps1 -download
    2) Avvia training:  .\quickstart.ps1 -train
    " $Colors.Success
}

# ─────────────────────────────────────────────────────────────
# DOWNLOAD: Scarica dataset ESIM
# ─────────────────────────────────────────────────────────────

function Invoke-Download {
    Write-Colored "
    ⬇️  DOWNLOAD DATASET
    ────────────────────" $Colors.Info

    Write-Colored "  📍 E-NeRF Dataset disponibili da:" $Colors.Info
    Write-Colored "     https://vision.in.tum.de/research/enerf" $Colors.Warning

    Write-Colored "  Dataset disponibili:" $Colors.Info
    Write-Colored "    - mocapDesk2 (~200 MB)     [CONSIGLIATO per primo test]" $Colors.Info
    Write-Colored "    - shakeMoon1 (~300 MB)" $Colors.Info
    Write-Colored "    - shakeCarpet1 (~250 MB)" $Colors.Info

    Write-Colored "
  � Link diretto (browser):" $Colors.Info
    Write-Colored "    mocapDesk2.tar:" $Colors.Warning
    Write-Colored "    https://vision.in.tum.de/research/enerf/mocapDesk2.tar" $Colors.Warning

    Write-Colored "
  📥 Opzione A: Download browser" $Colors.Info
    Write-Colored "    1) Apri il link qui sopra nel browser" $Colors.Info
    Write-Colored "    2) Salva mocapDesk2.tar in: $DATA_DIR\" $Colors.Info
    Write-Colored "    3) Estrai il file" $Colors.Info

    Write-Colored "
  📥 Opzione B: Download cmd (require tar)" $Colors.Warning
    @"
    # PowerShell
    cd "$DATA_DIR"
    # Su Windows 10 (build 17063+) o Windows 11, tar è built-in:
    curl.exe -o mocapDesk2.tar https://vision.in.tum.de/research/enerf/mocapDesk2.tar
    tar -xf mocapDesk2.tar
"@ | Write-Colored -Color $Colors.Info

    Write-Colored "
  ✅ Successivamente:" $Colors.Success
    Write-Colored "    Verifica: ls $DATA_DIR\mocapDesk2" $Colors.Info
    Write-Colored "    Deve contenere: images/, events/, *.npy, *.txt" $Colors.Success
}

# ─────────────────────────────────────────────────────────────
# TRAIN: Avvia training
# ─────────────────────────────────────────────────────────────

function Invoke-Train {
    Write-Colored "
    🚀 AVVIO TRAINING
    ──────────────────" $Colors.Info

    # Verifica dataset
    if (-not (Test-Path "$DATA_DIR\mocapDesk2\poses_bounds.npy")) {
        Write-Colored "  ❌ Dataset non trovato!" $Colors.Error
        Write-Colored "     Scarica prima: .\quickstart.ps1 -download" $Colors.Info
        exit 1
    }
    Write-Colored "  ✓ Dataset trovato" $Colors.Success

    # Config
    $configFile = "$REPO_ROOT\configs_custom\rtx4060ti_optimal.txt"
    Write-Colored "  📄 Config: $configFile" $Colors.Info

    # Parametri training
    Write-Colored "  
  ⚙️  Parametri Training:
     - Variant: FFMLP (veloce)
     - Mixed Precision (fp16): ON
     - Gradient Checkpointing: ON
     - Rays/iter: 3072
     - Steps/ray: 24+24
     - Iterations: 10000
     - Tempo stimato: 2-3 ore
     - Memory picco: ~5 GB
    " $Colors.Info

    Write-Colored "  Inizio training..." $Colors.Warning

    # Avvia training
    conda run -n $CONDA_ENV python main_nerf.py `
        --config $configFile `
        --gpu 0

    if ($LASTEXITCODE -eq 0) {
        Write-Colored "
    ✅ Training completato!
    
    Checkpoint salvato in:
    $EXP_DIR\enerf_rtx4060ti\
    
    Prossimi step:
    1) Visualizza loss:  .\quickstart.ps1 -status
    2) Genera output:    .\quickstart.ps1 -inference
    3) GUI interattiva:  .\quickstart.ps1 -gui
        " $Colors.Success
    } else {
        Write-Colored "  ❌ Training fallito!" $Colors.Error
        Write-Colored "     Veda troubleshooting in GUIDA_SETUP_LOCALE.md" $Colors.Warning
    }
}

# ─────────────────────────────────────────────────────────────
# INFERENCE: Genera output
# ─────────────────────────────────────────────────────────────

function Invoke-Inference {
    Write-Colored "
    📸 RENDERING INFERENZA
    ──────────────────────" $Colors.Info

    $expDir = "$EXP_DIR\enerf_rtx4060ti"
    $ckptFiles = Get-ChildItem "$expDir\ckpt\*.pth" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    
    if (-not $ckptFiles) {
        Write-Colored "  ❌ Checkpoint non trovato!" $Colors.Error
        Write-Colored "     Avvia training prima: .\quickstart.ps1 -train" $Colors.Info
        exit 1
    }

    $latestCkpt = $ckptFiles[0].FullName
    Write-Colored "  🔄 Usando checkpoint: $($ckptFiles[0].Name)" $Colors.Info

    $configFile = "$REPO_ROOT\configs_custom\rtx4060ti_optimal.txt"

    Write-Colored "
  📥 Modalità rendering:" $Colors.Info
    Write-Colored "    [1] Keyframe poses (da training set)" $Colors.Info
    Write-Colored "    [2] Random interpolated poses" $Colors.Info
    Write-Colored "    [3] Estrai mesh 3D (per Blender/MeshLab)" $Colors.Info
    Write-Colored "    [4] Annulla" $Colors.Info

    $choice = Read-Host "  Scelta (1-4)"

    switch ($choice) {
        "1" {
            Write-Colored "  Rendering keyframe poses..." $Colors.Info
            conda run -n $CONDA_ENV python main_nerf.py `
                --config $configFile `
                --ckpt $latestCkpt `
                --test_only `
                --gpu 0
        }
        "2" {
            Write-Colored "  Rendering random poses (60 frame)..." $Colors.Info
            conda run -n $CONDA_ENV python scripts\render.py `
                --model_dir $expDir `
                --rand_poses 1 `
                --num_poses 60 `
                --output_dir "$expDir\results_random"
        }
        "3" {
            Write-Colored "  Estraendo mesh..." $Colors.Info
            conda run -n $CONDA_ENV python main_nerf.py `
                --config $configFile `
                --ckpt $latestCkpt `
                --test_only `
                --save_mesh 1 `
                --mesh_resolution 256
        }
        default {
            Write-Colored "  Annullato." $Colors.Warning
            return
        }
    }

    Write-Colored "
    ✅ Inferenza completata!
    Output in: $expDir\results\
    " $Colors.Success
}

# ─────────────────────────────────────────────────────────────
# GUI: Avvia GUI interattiva
# ─────────────────────────────────────────────────────────────

function Invoke-GUI {
    Write-Colored "
    🖼️  GUI INTERATTIVA
    ──────────────────" $Colors.Info

    $expDir = "$EXP_DIR\enerf_rtx4060ti"
    $ckptFiles = Get-ChildItem "$expDir\ckpt\*.pth" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    
    if (-not $ckptFiles) {
        Write-Colored "  ❌ Checkpoint non trovato!" $Colors.Error
        exit 1
    }

    $latestCkpt = $ckptFiles[0].FullName
    $configFile = "$REPO_ROOT\configs_custom\rtx4060ti_optimal.txt"

    Write-Colored "  🖱️  Controlli:" $Colors.Info
    Write-Colored "     Mouse drag: Rotazione camera" $Colors.Info
    Write-Colored "     Scroll: Zoom" $Colors.Info
    Write-Colored "     W/A/S/D: Movimento" $Colors.Info
    Write-Colored "     +/-: FOV" $Colors.Info
    Write-Colored "     H: Help" $Colors.Info
    Write-Colored "     Q: Esci" $Colors.Info

    conda run -n $CONDA_ENV python main_nerf.py `
        --config $configFile `
        --ckpt $latestCkpt `
        --gui `
        --gpu 0
}

# ─────────────────────────────────────────────────────────────
# STATUS: Monitora training
# ─────────────────────────────────────────────────────────────

function Invoke-Status {
    Write-Colored "
    📊 TRAINING STATUS
    ──────────────────" $Colors.Info

    # GPU Stats
    Write-Colored "  🎮 GPU Stats:" $Colors.Info
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.free `
        --format=csv,noheader | ForEach-Object {
        Write-Colored "     $_" $Colors.Info
    }

    # Experiment status
    Write-Colored "
  🔬 Esperimenti:" $Colors.Info
    
    if (Test-Path "$EXP_DIR\enerf_rtx4060ti") {
        $ckptCount = @(Get-ChildItem "$EXP_DIR\enerf_rtx4060ti\ckpt\*.pth" -ErrorAction SilentlyContinue).Count
        Write-Colored "     ✓ enerf_rtx4060ti ($ckptCount checkpoints)" $Colors.Success
        
        if (Test-Path "$EXP_DIR\enerf_rtx4060ti\events.out*") {
            Write-Colored "
  📈 Visualizza training live (Tensorboard):" $Colors.Info
            Write-Colored "     tensorboard --logdir=exp/enerf_rtx4060ti" $Colors.Warning
            Write-Colored "     Apri: http://localhost:6006" $Colors.Info
        }
    } else {
        Write-Colored "     ❌ Nessun esperimento trovato" $Colors.Error
    }

    # Dataset
    Write-Colored "
  📂 Dataset:" $Colors.Info
    if (Test-Path "$DATA_DIR\mocapDesk2") {
        Write-Colored "     ✓ mocapDesk2 trovato" $Colors.Success
    } else {
        Write-Colored "     ❌ Dataset non trovato - scarica con: .\quickstart.ps1 -download" $Colors.Error
    }
}

# ─────────────────────────────────────────────────────────────
# MAIN: Dispatcher
# ─────────────────────────────────────────────────────────────

switch ($Action) {
    "setup" { Invoke-Setup }
    "train" { Invoke-Train }
    "inference" { Invoke-Inference }
    "gui" { Invoke-GUI }
    "status" { Invoke-Status }
    "download" { Invoke-Download }
    default {
        Write-Colored "
  Uso: .\quickstart.ps1 -action <action>
  
  Azioni disponibili:
    setup       Configura environment Conda e moduli CUDA
    download    Scarica dataset ESIM
    train       Avvia training
    inference   Genera output rendering
    gui         Avvia GUI interattiva
    status      Monitora training
  
  Esempio:
    .\quickstart.ps1 -setup
    .\quickstart.ps1 -train
    .\quickstart.ps1 -status
        " $Colors.Info
    }
}

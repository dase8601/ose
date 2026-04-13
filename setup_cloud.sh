#!/bin/bash
# setup_cloud.sh — One-shot setup for RunPod / Colab / Alpine (CUDA GPU)
#
# Usage:
#   bash setup_cloud.sh           # DoorKey/Crafter (system python)
#   source setup_cloud.sh habitat # Habitat (creates conda env with Python 3.9)
#
# Then run:
#   DoorKey:  python abm_experiment.py --all --device auto --steps 800000
#   Crafter:  python abm_experiment.py --all --device auto --env crafter --steps 1000000
#   Habitat:  python abm_experiment.py --all --device auto --env habitat --steps 500000
#
# Habitat requires A100 (80GB) for V-JEPA 2.1 ViT-B inference.
# habitat-sim only supports Python <=3.9, so we create a dedicated conda env.
#
# On Google Colab, mount Drive first and cd to your repo:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/jepa

set -e

HABITAT_MODE="${1:-}"

# ---------------------------------------------------------------------------
# Habitat mode: conda env with Python 3.9 (habitat-sim requirement)
# ---------------------------------------------------------------------------
if [ "$HABITAT_MODE" = "habitat" ]; then
    echo "=== Habitat mode: setting up conda env with Python 3.9 ==="

    # Install miniconda if not present
    if ! command -v conda &>/dev/null; then
        echo "Installing miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p /opt/conda
        export PATH="/opt/conda/bin:$PATH"
        rm /tmp/miniconda.sh

        # Accept TOS non-interactively
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    fi
    export PATH="/opt/conda/bin:$PATH"

    # Create Python 3.9 env (habitat-sim max supported version)
    if ! conda env list | grep -q "habitat"; then
        echo "Creating conda env 'habitat' with Python 3.9..."
        conda create -n habitat python=3.9 -y
    fi

    # Activate (works when script is sourced)
    source /opt/conda/bin/activate habitat

    echo "=== Installing habitat-sim ==="
    conda install habitat-sim headless -c conda-forge -c aihabitat -y

    echo "=== Installing PyTorch (CUDA 12.1) ==="
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

    echo "=== Installing core + V-JEPA dependencies ==="
    pip install gymnasium numpy matplotlib omegaconf timm minigrid crafter -q

    echo "=== Verifying (habitat mode) ==="
    python -c "
import torch, gymnasium
print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')

import habitat_sim; print('habitat-sim: OK')

print()
print('Ready! Run:')
print('  python abm_experiment.py --all --device auto --env habitat --steps 500000')
"
    echo ""
    echo "=== IMPORTANT: conda env 'habitat' is now active ==="
    echo "=== If you open a new terminal, run: source /opt/conda/bin/activate habitat ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Standard mode: DoorKey / Crafter (system python, no conda needed)
# ---------------------------------------------------------------------------
echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Installing core dependencies ==="
pip install minigrid gymnasium numpy matplotlib crafter -q

# V-JEPA 2.1 dependencies (needed if using vjepa encoder on non-habitat envs)
pip install omegaconf timm -q 2>/dev/null || true

echo "=== Verifying ==="
python -c "
import torch, gymnasium, minigrid
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')
print(f'gymnasium: {gymnasium.__version__}')
print('minigrid: OK')

try:
    import crafter; print('crafter: OK')
except: print('crafter: not installed')

print()
print('Run commands:')
print('  DoorKey: python abm_experiment.py --all --device auto --steps 800000')
print('  Crafter: python abm_experiment.py --all --device auto --env crafter --steps 1000000')
print('  Habitat: source setup_cloud.sh habitat  (needs separate setup)')
"

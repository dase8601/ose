#!/bin/bash
# setup_cloud.sh — One-shot setup for RunPod / Colab / Alpine (CUDA GPU)
#
# Usage:
#   bash setup_cloud.sh
#
# Then run:
#   python abm_experiment.py --all --device auto --steps 400000
#
# On Google Colab, mount Drive first and cd to your repo:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/jepa

set -e

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Installing experiment dependencies ==="
pip install minigrid gymnasium numpy matplotlib crafter -q

echo "=== Verifying ==="
python -c "
import torch, gymnasium, minigrid, crafter
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'gymnasium: {gymnasium.__version__}')
print('minigrid: OK')
print('crafter: OK')
print('All good!')
print('  DoorKey: python abm_experiment.py --all --device auto --steps 800000')
print('  Crafter: python abm_experiment.py --all --device auto --env crafter --steps 1000000')
"

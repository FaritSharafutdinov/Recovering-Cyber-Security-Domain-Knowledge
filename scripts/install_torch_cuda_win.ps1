# Install PyTorch with CUDA on Windows (Python 3.14: stable cu12x wheels may be missing — use nightly cu128).
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/install_torch_cuda_win.ps1
$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Write-Host "Python launcher 'py' not found. Install Python 3.14 or set PATH to python.exe" -ForegroundColor Red
    exit 1
}
& $py -3.14 -m pip uninstall torch torchvision torchaudio -y
& $py -3.14 -m pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
& $py -3.14 -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to PyTorch'; print(torch.__version__, torch.cuda.get_device_name(0))"

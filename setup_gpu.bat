@echo off
echo ============================================================
echo  GPU Setup for MIMIC Fund Manager KG -- RTX 3070 CUDA 12.1
echo ============================================================

echo Installing PyTorch 2.1 with CUDA 12.1...
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo Installing PyTorch Geometric...
pip install torch-geometric

echo Installing sparse/scatter extensions (CUDA 12.1)...
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

echo Installing sklearn for metrics...
pip install scikit-learn numpy pandas

echo.
echo ============================================================
echo  Verify installation:
echo ============================================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "from torch_geometric.nn import HGTConv; print('PyG: OK')"

echo.
echo ============================================================
echo  Setup complete. Run:
echo    python step13b_rgcn.py
echo ============================================================
pause

# AutoRig – Python Environment Setup

This project uses a dedicated Conda environment called `netflix_f25` to manage Python dependencies.

## 1. Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python compatible with PyTorch (e.g. 3.10)

## 2. Create and Activate the Conda Environment

Open a terminal (or Anaconda Prompt on Windows) and run:

```bash
# Create environment (you can change the Python version if needed)
conda create -n netflix_f25 python=3.10 -y

# Activate the environment
conda activate netflix_f25
```

## 3. Install dependency
```bash
pip install torch torchvision opencv-python pyyaml numpy
pip install ultralytics
pip install huggingface_hub
pip install imageio
pip install scipy
pip install shapely
pip install scikit-image
pip install pygltflib
```

## 4. Download Pretrained Pose Estimation Weights
1. Open this Google Drive folder in your browser: https://drive.google.com/drive/folders/1fGLeCgTbaO50wylfV_j1OFTEx8DDpaqh
2. Download the file: pose_resnet_50_256x192.pth.tar
3. Create a folder called pretrained_weights in Assets/Scripts
4. Save the file in the Assets/Scripts/pretrained_weights folder 


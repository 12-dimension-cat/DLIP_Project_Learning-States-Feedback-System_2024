# Anaconda Installation and Environment Setup Guide

## 1. Installing Anaconda and Creating an Environment

### Install Anaconda
If Anaconda is not installed, download the installer from the [Anaconda download page](https://www.anaconda.com/products/distribution) and install it.

### Create a Virtual Environment
Open the Anaconda prompt and enter the following command to create a virtual environment:
```bash
conda create -n myenv python=3.9.18
```
`myenv` is the name of the virtual environment. You can change it to any desired name.

## 2. Activate the Virtual Environment
Activate the created virtual environment.
```bash
conda activate myenv
```

## 3. Install Required Packages

### Install numpy
```bash
pip install numpy==1.26.4
```

### Install OpenCV
```bash
pip install opencv-python==4.10.0
```

## 4. Install PyTorch and CUDA

### Check Graphics Card and CUDA Compatibility

#### Check NVIDIA Driver Installation and Version
To check if CUDA is available, an NVIDIA graphics card must be installed. Use the following command to check:
```bash
nvidia-smi
```
This command shows the currently installed NVIDIA driver and CUDA version. If the `nvidia-smi` command does not work, the NVIDIA driver may not be installed or an NVIDIA graphics card may not be present.

#### Check CUDA Compatibility
Visit the [CUDA compatibility page](https://developer.nvidia.com/cuda-gpus) to check which CUDA version your graphics card supports.

### Install PyTorch and CUDA
To install PyTorch, refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) Get Started page for the appropriate command. For example, to install PyTorch 2.1.2 with CUDA 11.8:
```bash
pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 5. Install YOLOv8 and YOLOv8-pose
YOLOv8 and YOLOv8-pose can be installed from the Ultralytics YOLOv8 repository. First, clone it via git and then install it.
```bash
# Install Yolov8
pip install ultralytics

# Install Yolov8-pose (yolov8-pose has extended features from yolov8, so additional packages may be required.)
pip install -U ultralytics[pose]
```

## 6. Verify Installation
To ensure all packages are installed correctly, use the following commands to print the version of each package.
```bash
python -c "import numpy as np; print(np.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
```

## 7. Start Working in the Virtual Environment
Perform the required tasks in the activated virtual environment. Install additional packages or change settings as needed.

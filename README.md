1. Install Anaconda and Create a Virtual Environment
Install Anaconda
If Anaconda is not already installed, download and install it from the Anaconda Distribution page.

Create a Virtual Environment
Open the Anaconda Prompt and create a new virtual environment with Python 3.9.18:

bash
코드 복사
conda create -n myenv python=3.9.18
myenv is the name of the virtual environment. You can change it to any name you prefer.

2. Activate the Virtual Environment
Activate the virtual environment:

bash
코드 복사
conda activate myenv
3. Install Required Packages
Install numpy
bash
코드 복사
pip install numpy==1.26.4
Install OpenCV
bash
코드 복사
pip install opencv-python==4.10.0
4. Install PyTorch with CUDA
Check Graphics Card and CUDA Compatibility
First, ensure that an NVIDIA graphics card is installed and that CUDA is supported. Use the following command to check for an NVIDIA GPU:

bash
코드 복사
nvidia-smi
This command shows the installed NVIDIA driver and CUDA version. If the nvidia-smi command does not work, it means either the NVIDIA driver is not installed or there is no NVIDIA graphics card.

Verify CUDA Compatibility
Visit the CUDA GPUs page to check which CUDA version your GPU supports.

Install PyTorch with CUDA
To install PyTorch 2.1.2 with CUDA 11.8, use the command provided on the PyTorch Get Started page. For example:

bash
코드 복사
pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
5. Install YOLOv8 and YOLOv8-pose
YOLOv8 and YOLOv8-pose can be installed from the Ultralytics YOLOv8 repository. Install them using pip:

bash
코드 복사
# Install YOLOv8
pip install ultralytics

# Install YOLOv8-pose (YOLOv8-pose extends YOLOv8 functionality, so additional packages might be required)
pip install -U ultralytics[pose]
6. Verify Installation
Ensure all packages are correctly installed by checking their versions:

python
코드 복사
python -c "import numpy as np; print(np.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
7. Start Working in the Virtual Environment
With the virtual environment activated, you can proceed with your project. Install any additional packages or configure settings as needed.1. Install Anaconda and Create a Virtual Environment
Install Anaconda
If Anaconda is not already installed, download and install it from the Anaconda Distribution page.

Create a Virtual Environment
Open the Anaconda Prompt and create a new virtual environment with Python 3.9.18:

bash
코드 복사
conda create -n myenv python=3.9.18
myenv is the name of the virtual environment. You can change it to any name you prefer.

2. Activate the Virtual Environment
Activate the virtual environment:

bash
코드 복사
conda activate myenv
3. Install Required Packages
Install numpy
bash
코드 복사
pip install numpy==1.26.4
Install OpenCV
bash
코드 복사
pip install opencv-python==4.10.0
4. Install PyTorch with CUDA
Check Graphics Card and CUDA Compatibility
First, ensure that an NVIDIA graphics card is installed and that CUDA is supported. Use the following command to check for an NVIDIA GPU:

bash
코드 복사
nvidia-smi
This command shows the installed NVIDIA driver and CUDA version. If the nvidia-smi command does not work, it means either the NVIDIA driver is not installed or there is no NVIDIA graphics card.

Verify CUDA Compatibility
Visit the CUDA GPUs page to check which CUDA version your GPU supports.

Install PyTorch with CUDA
To install PyTorch 2.1.2 with CUDA 11.8, use the command provided on the PyTorch Get Started page. For example:

bash
코드 복사
pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
5. Install YOLOv8 and YOLOv8-pose
YOLOv8 and YOLOv8-pose can be installed from the Ultralytics YOLOv8 repository. Install them using pip:

bash
코드 복사
# Install YOLOv8
pip install ultralytics

# Install YOLOv8-pose (YOLOv8-pose extends YOLOv8 functionality, so additional packages might be required)
pip install -U ultralytics[pose]
6. Verify Installation
Ensure all packages are correctly installed by checking their versions:

python
코드 복사
python -c "import numpy as np; print(np.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
7. Start Working in the Virtual Environment
With the virtual environment activated, you can proceed with your project. Install any additional packages or configure settings as needed.

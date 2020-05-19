# The Effects of Different Levels of Realism (IJCARS2020)

# Citation
This code refers to the following publication. 
```bibtex
@Article{herediaperez2020effects,
  author       = {Heredia Perez, S. A. and Marinho, M. M. and Harada, K. and Mitsuishi, Mamoru},
  title        = {The Effects of Different Levels of Realism on the Training of CNNs with only Synthetic Images for the Semantic Segmentation of Robotic Instruments in a Head Phantom},
  journal      = {International Journal of Computer Assisted Radiology and Surgery (IJCARS)},
  year         = {2020},
  doi          = {10.1007/s11548-020-02185-0},
}
```
# Installation
This installation has been tested on Windows 10 64 bit.

## Hardware requirements
This code was tested on a NVIDIA RTX 2070. The code will automatically assign one model for each available GPU.

## Python 3.7 
Download Python 3.7 x64
https://www.python.org/downloads/windows/

## Tensorflow 2.1.0 GPU
Tensorflow 2.1.0 GPU has the following requirements.
- NVIDIA GPU drivers - 418.x or higher.
- CUDA Toolkit - CUDA 10.1
- cuDNN SDK = 7.6

For more information, refer to
https://www.tensorflow.org/install/gpu

## Other requirements
A virtual environment is recommended. After setting up your virtual environment, run
```shell script
python3 -m pip install -r requirements.txt
```

## Download synthetic image database
TODO

## Run program
```shell script
python3 main.py
```
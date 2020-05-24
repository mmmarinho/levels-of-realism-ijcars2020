# The Effects of Different Levels of Realism (IJCARS2020)

# License
This code is GPLv3 licensed. Please read and understand the terms of the license before using (or reading) the code available in this repository.
 
A couple of useful links on this topic.
- https://www.gnu.org/licenses/gpl-3.0.en.html
- https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)

# Citation
Cite the following publication.
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
Download and install Python 3.7 x64
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
The data is available at IEEE DataPort
http://dx.doi.org/10.21227/xmc2-1v59

Download and extract `data.zip` and `validation_data.zip`

### `data.zip`
Contains 10376 synthetically generated images for each renderer and corresponding automatically-annotated ground-truths. 

|Folder|Meaning|
|---|---|
|data/1_flat_renderer/image|Images rendered with the flat renderer|
|data/1_flat_renderer/label|Flat renderer ground-truth|
|data/2_basic_renderer/image|Images rendered with the basic renderer|
|data/2_basic_renderer/label|Basic renderer ground-truth|
|data/3_realistic_renderer/image|Images rendered with the realistic renderer|
|data/3_realistic_renderer/label|Realistic renderer ground-truth|

### `validation_data.zip`
|Folder|Meaning|
|---|---|
|validation_data/image|Images obtained from the physical SmartArm setup|
|validation_data/label|Manually-annotated ground-truth|


## Run 
```shell script
python3 main.py
```

## Results
Output images during training and trained models will be saved to the corresponding `output` folder of each renderer.

## Configuration
Most parameters relevant for training can be modified in the `configuration.yml` file. 

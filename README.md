# General Deep Image Completion with Lightweight cGANs
These are the implementations and simple demo system of my bmvc'17 paper: General Deep Image Completion with Lightweight cGANs. The main goal of our paper is to build deep image completion models for recovering various corrupted images, including

- General Image Completion

The objective of this model is to complete images with differnet types of corrupted masks like scribbles, lines, dots and texts.

![general image completion](https://raw.githubusercontent.com/adamstseng/general-deep-image-completion/master/general-completion.jpg)

- Face Completion

This is an extension of our general image completion model that not only tackles the face completion task but also aiming on recovering faces with arbitrary missing regions. We apply a differnet but stronger deep autoencoder structure in this model.

![face completion](https://raw.githubusercontent.com/adamstseng/general-deep-image-completion/d8211e9434af07b74bda3f304c16658189c380fc/face-completion.jpg)

## Training 
TBA

## Demo System
We build a simple interactive demonstration of image completion based on Python, OpenCV and TensorFlow on Windows

### Videos (Youtube Links)

[General Image Completion](https://www.youtube.com/watch?v=513xQM4NrxY&feature=youtu.be) 

[Face Completion](https://www.youtube.com/watch?v=MWj2kkMDrgY&feature=youtu.be)

### Setups
- Install [Visual Studio Code](https://code.visualstudio.com/): A source code editor (Optional, but highly recommended)
- Follow the guidelines from [TensorFlow](https://www.tensorflow.org/install/install_windows) to instal Python 3.5.2 and TensorFlow on Windows
- Install libraries numpy and opencv-python through pip
- Download [face-completion-model](https://drive.google.com/file/d/0BwBvCjzIsl2vZHoxS0RrRm55d1U/view?usp=sharing) and [general-completion-model](https://drive.google.com/file/d/0BwBvCjzIsl2vV3FvZUd0VjdxZE0/view?usp=sharing). Put them into corresponding folders
- Run **demo.py** in general-image-completion or face-completion folder

If you fail to import tensorflow, please refer to [this article](https://github.com/tensorflow/tensorflow/issues/8385) for possible solutions.

### Controls
- Moving mouse to draw scribbles on given images
- Key c: convert corrupted images to completed images
- Key r: resume any scribbles
- Key n: next image

## Citation


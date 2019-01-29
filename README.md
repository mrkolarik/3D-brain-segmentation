# Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation
Hello everyone, this is a repository containing code to Paper "Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation" published at MDPI Applied sciences journal - https://www.mdpi.com/2076-3417/9/3/404 .

Most useful parts of this repository are python keras scripts with source code for 2D and 3D Dense-U-Net network models. Its an upgraded U-Net that we obtained better results than with classic U-Net and current hardware is capable of fitting it into GPU memory. Happy experimenting and let me know if any of your work is inspired by our work :) !

Please cite our work as:

Kolařík, M., Burget, R., Uher, V., Říha, K., & Dutta, M. K. (2019). Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation. Applied Sciences, 9(3), vol. 9, no. 3. 


Article{kolavrik2019optimized,<br>
    title={Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation},<br>
    author={Kola{\v{r}}{\'\i}k, Martin and Burget, Radim and Uher, V{\'a}clav and {\v{R}}{\'\i}ha, Kamil and Dutta, Malay Kishore},<br>
    journal={Applied Sciences},
    volume={9},<br>
    number={3},<br>
    pages={404},<br>
    year={2019},<br>
    publisher={Multidisciplinary Digital Publishing Institute} <br>
}


The Code is inspired by great repository [Deep Learning Tutorial for Kaggle Ultrasound Nerve Segmentation competition, using Keras](https://github.com/jocicmarko/ultrasound-nerve-segmentation)
The Dense-Unet architecture was inspired by papers [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf) and [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf).
The original U-Net architecture was inspired by paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


## Overview

<p align="center">
  <img height="300" src="img/combination.png"> <br>
<b>Figure_1:</b> Example of MRI sagitall brain scan slice and transversal thoracic CT scan slice - tissue segmented with 3D-Dense-U-Net is highlighted in yellow.
</p>

<p align="center">
  <img height="450" src="img/unet_final.png"> <br>
<b>Figure_2:</b> Dense-U-net network model. Residual interconnections are in green color, dense interconnections in blue.
</p>

<p align="center">
  <img height="300" src="img/dense_brain.png"> <br>
<b>Figure_3:</b> Brain model segmented from MRI set of images by 3D-Dense-U-Net.
</p>

<p align="center">
  <img height="500" src="img/dense_spine.png"> <br>
<b>Figure_4:</b> Spine model segmented from CT set of images by 3D-Dense-U-Net. The abnormal vertebrae adhesions exist also in original ground truth masks.
</p>


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.

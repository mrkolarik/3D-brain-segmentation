# Optimized High Resolution 3D Dense-U-Net for Brain and Spine Segmentation

<p align="center">
  <a href="https://www.mdpi.com/2076-3417/9/3/404"><img src="https://img.shields.io/badge/Paper-Applied%20Sciences%20(2019)-blue"></a>
  <img src="https://img.shields.io/badge/Status-To%20Be%20Archived-lightgrey">
  <img src="https://img.shields.io/badge/Framework-Keras-orange">
</p>

---

This repository provides the official codebase for the paper:  
**[Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation](https://www.mdpi.com/2076-3417/9/3/404)**  
published in *MDPI Applied Sciences, 2019*.

If you are working on medical 3D segmentation and seek to benchmark against our methods, this repository offers the corresponding academic implementation.

---

## 📄 Paper Overview

- Optimized Dense-U-Net architecture for brain MRI and spine CT segmentation
- Improvements over standard U-Net in segmentation accuracy
- Designed for practical training on available GPUs in 2019
- Implementations include:
  - 2D and 3D Dense-U-Net
  - 2D and 3D Res-U-Net
  - Classic U-Net for baseline comparison

---

## 📂 Datasets used

- **Spine Dataset:** Previously available through [SpineWeb](https://spineweb.digitalimaginggroup.ca/). Offline as of April 2025
- **Brain MRI Dataset:** Private; redistribution is unfortunately not permitted.

---

## 🧪 Results

Segmentation examples using 3D Dense-U-Net:

| Brain MRI Slice | Dense-U-Net Architecture | Brain Segmentation Output | Spine Segmentation Output |
|:---:|:---:|:---:|:---:|
| ![](img/combination.png) | ![](img/unet_final.png) | ![](img/dense_brain.png) | ![](img/dense_spine.png) |

- **Figure 1:** Example of MRI brain slice and thoracic CT scan. Segmented tissue is highlighted.
- **Figure 2:** Dense-U-Net architecture (green: residual connections, blue: dense connections).
- **Figure 3:** 3D brain segmentation output.
- **Figure 4:** 3D spine segmentation output (vertebrae abnormalities included in ground truth).

---

## 📚 Citation

If you reference or build on this work, please cite:

```
@article{kolarik2019optimized,
  title={Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation},
  author={Kola{\v{r}}{\'\i}k, Martin and Burget, Radim and Uher, V{\'a}clav and {\v{R}}{\'\i}ha, Kamil and Dutta, Malay Kishore},
  journal={Applied Sciences},
  volume={9},
  number={3},
  pages={404},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

---

## 📖 Related Work

This implementation is inspired by:

- [Deep Learning Tutorial for Kaggle Ultrasound Nerve Segmentation](https://github.com/jocicmarko/ultrasound-nerve-segmentation)
- [The One Hundred Layers Tiramisu (Fully Convolutional DenseNets)](https://arxiv.org/pdf/1611.09326.pdf)
- [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/pdf/1608.06993.pdf)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

---

## 🚀 Project Status

This repository is no longer actively maintained and will be archived.  
The code serves as an academic reference for reproducing results from the original publication.  
For current state-of-the-art (SOTA) research, please consult newer architectures and frameworks.

---

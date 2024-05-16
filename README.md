# Image Fusion Project

This repository contains a collection of scripts for various image fusion techniques, including VGG19 feature extraction, Laplacian pyramid fusion, PCA fusion, and DWT fusion. Additionally, a simple GUI for image fusion is provided.

## Introduction

The Image Fusion Project aims to combine different image fusion techniques to create a comprehensive toolkit for image fusion tasks. This is particularly useful in medical imaging where CT and MRI scans need to be fused to provide better diagnostic information. The project includes implementations of the following methods:

- VGG19-based feature extraction
- Laplacian pyramid fusion
- PCA-based fusion
- Discrete Wavelet Transform (DWT) fusion

A graphical user interface (GUI) is also provided to facilitate easy use of these fusion methods.

## Features

- **VGG19 Feature Extraction**: Uses a pre-trained VGG19 model to extract deep features from images.
- **Laplacian Pyramid Fusion**: Combines images using multi-resolution analysis for enhanced detail preservation.
- **PCA Fusion**: Applies Principal Component Analysis to fuse images by transforming them into a new feature space.
- **DWT Fusion**: Uses Discrete Wavelet Transform to combine images based on their frequency components.
- **GUI**: Provides a user-friendly interface for selecting and fusing images using the above methods.

## Requirements

- Python 3.6+
- TensorFlow
- Keras
- OpenCV
- Numpy
- Scikit-learn
- PyWavelets
- Pillow
- Tkinter

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/bharath9812/Image_Fusion
   cd image-fusion
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## How to Use

To start the GUI and use the image fusion methods, ensure all the required scripts and modules are in the same directory. Then, run the `gui.py` script:

```sh
python gui.py
```

Follow the instructions on the GUI to select and fuse images. The GUI supports simple image fusion methods like blending.

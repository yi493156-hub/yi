
# Face Emotion Recognition Project


This project implements a real-time facial emotion recognition system using Convolutional Neural Networks (CNN) and a local camera feed.


##  Project Structure

* **project code/**: Contains scripts for data processing, Colab training notebooks, and the main real-time recognition program.

* **trained model/**: Directory intended for model weights (Note: Large files like .pkl/.pth are excluded due to GitHub size limits).

* **.gitignore**: Configuration to prevent large binaries and temporary files from being uploaded.


## Quick Start

1. **Prerequisites**: Ensure you have Python installed with libraries such as `opencv-python`, `tensorflow`, or `pytorch`.

2. **Run Recognition**: Execute the main script:

   ```bash

   python "project code/Face Emotion Recognition Using Local Camera.py"

   ```


##  Experiments

The project includes comparative studies on various hyperparameters including:

* Batch Sizes (64 vs 256)

* Optimizers (SGD vs others)

* Regularization (L2, Dropout)

* Model Architecture (ResBlock variations)


## Author

Developed as part of the Final Project for the Deep Learning course.


## 📺 Project Demo
Watch the real-world emotion recognition in action:

https://github.com/yi493156-hub/yi/raw/main/demo.mp4

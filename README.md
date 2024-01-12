# AMLS_assignment23_24_SN20049328

## Project Description
This project is focused on applying Convolutional Neural Networks (CNNs) to medical image classification. It consists of two key tasks:
- **Task A**: Binary classification of chest X-ray images for pneumonia detection.
- **Task B**: Multi-class classification of different tissue types in histopathological images.

## Organization
The project is structured into three main directories:

- `A/`: Contains code related to the pneumonia classification task (Task A).
- `B/`: Contains code related to the tissue type classification task (Task B).
- `Datasets/`: Stores the PneumoniaMNIST and PathMNIST datasets used for testing the models in Task A and Task B, respectively.

Execute `main.py` to test Task A and Task B individually. To focus on testing one task at a time, comment out the section of the code corresponding to the other task.

## Required Packages
To run the code, the following Python packages are required:
- TensorFlow: An open-source machine learning library for building and training neural networks.
- NumPy: A library for large, multi-dimensional array and matrix processing.
- Matplotlib: A plotting library for creating static, animated, and interactive visualizations.
- Seaborn: A Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

## Running Environment
The project has been developed and tested in the following environment:

- **Python Version**: `3.9.13`
- **Key Dependencies**:
  - TensorFlow: `2.15.0`
  - NumPy: `1.24.4`
  - Matplotlib: `3.5.2`
  - Seaborn: `0.11.2`

# Generative Adversarial Networks (GANs) in PyTorch

A complete implementation of a Generative Adversarial Network (GAN) using PyTorch, designed to run on Google Colab.

## Overview

This repository contains a fully functional GAN implementation that generates handwritten digits similar to the MNIST dataset. The implementation includes:

- **Generator Network**: Creates fake images from random noise
- **Discriminator Network**: Distinguishes between real and fake images
- **Training Loop**: Adversarial training process
- **Visualization**: Tools to visualize generated images and training progress

## Features

- ✅ Complete GAN implementation in PyTorch
- ✅ Optimized for Google Colab (GPU support)
- ✅ MNIST dataset integration
- ✅ Training visualization and loss plotting
- ✅ Model saving and loading
- ✅ Comprehensive documentation and comments

## Quick Start

### Running on Google Colab (Recommended)

1. Open the notebook in Google Colab:
   - Upload `GAN_PyTorch.ipynb` to your Google Drive
   - Open it with Google Colab
   - Or directly: [Open in Colab](https://colab.research.google.com/)

2. Enable GPU acceleration:
   - Go to `Runtime` → `Change runtime type`
   - Select `GPU` as Hardware accelerator
   - Click `Save`

3. Run all cells:
   - Click `Runtime` → `Run all`
   - Or use `Ctrl+F9`

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/oloketuyimoyosi/Generative-Adversarial-Networks-GANs-.git
cd Generative-Adversarial-Networks-GANs-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook GAN_PyTorch.ipynb
```

## What You'll Learn

- How GANs work and the adversarial training process
- Implementing neural networks in PyTorch
- Training deep learning models
- Generating synthetic images
- Visualizing training progress

## Requirements

- Python 3.6+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- matplotlib 3.3.4+
- numpy 1.19.5+

See `requirements.txt` for complete dependencies.

## Architecture

### Generator
- Input: Random noise vector (100 dimensions)
- Hidden layers: 256 → 512 → 1024 neurons
- Output: 784 dimensions (28×28 image)
- Activation: LeakyReLU, Tanh (output)

### Discriminator
- Input: Flattened image (784 dimensions)
- Hidden layers: 1024 → 512 → 256 neurons
- Output: Single probability (real/fake)
- Activation: LeakyReLU, Sigmoid (output)

## Training Details

- **Dataset**: MNIST (60,000 training images)
- **Batch Size**: 64
- **Epochs**: 50
- **Optimizer**: Adam (lr=0.0002)
- **Loss Function**: Binary Cross-Entropy

## Results

After training, the Generator can create realistic handwritten digits. The notebook includes visualization tools to:
- Compare real vs generated images
- Plot training losses over time
- Display generated samples

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## Author

Created with ❤️ for learning and exploring GANs in PyTorch

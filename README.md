# Vision Transformer (ViT) From Scratch – PyTorch

A simple implementation of the **Vision Transformer (ViT)** built from scratch using **PyTorch**.
This project focuses on understanding the internal components of transformers applied to images, including **patch embeddings, multi-head self-attention, positional embeddings, and transformer encoder blocks**.

The goal of this project is educational: to understand how Vision Transformers work internally rather than relying on high-level libraries.

---

## Overview

Traditional Convolutional Neural Networks (CNNs) process images using convolution operations.
Vision Transformers instead treat an image as a **sequence of patches**, similar to how transformers process tokens in NLP.

Pipeline:

```
Image
  ↓
Patch Embedding (Conv2d)
  ↓
Flatten + Token Embeddings
  ↓
Add CLS Token
  ↓
Add Positional Embeddings
  ↓
Transformer Encoder Blocks
  ↓
Classification Head
```

---

## Features

* Vision Transformer implemented **from scratch**
* Patch extraction using **Conv2d**
* Learnable **CLS Token**
* Learnable **Positional Embeddings**
* **Multi-Head Self Attention**
* **Layer Normalization**
* **Feed Forward Network (MLP)**
* Residual connections
* Training loop implemented in PyTorch

---

## Project Structure

```
vision-transformer/
│
├── model.py        # Vision Transformer architecture
├── train.py        # Training loop
├── dataset.py      # Data loading utilities
├── utils.py        # Helper functions
├── README.md
```

---

## Model Architecture

### 1. Patch Embedding

Images are divided into patches using a convolution layer:

```
Conv2d(
  in_channels = 3,
  out_channels = embed_dim,
  kernel_size = patch_size,
  stride = patch_size
)
```

Example:

```
Input Image: (B, 3, 32, 32)
Output:      (B, embed_dim, num_patches)
```

---

### 2. CLS Token

A learnable **classification token** is added to the beginning of the token sequence.

```
[CLS] Patch1 Patch2 Patch3 ... PatchN
```

The final classification prediction is taken from the CLS token representation.

---

### 3. Positional Embeddings

Transformers do not understand spatial relationships by default, so positional embeddings are added:

```
x = x + positional_embedding
```

This helps the model learn the spatial arrangement of patches.

---

### 4. Multi-Head Self Attention

Multi-head attention allows the model to attend to different relationships between patches.

Steps:

```
1. Compute Q, K, V
2. Split embedding into multiple heads
3. Compute attention scores
4. Apply softmax
5. Weighted sum with V
6. Merge heads
```

Mathematically:

```
Attention(Q,K,V) = softmax(QKᵀ / √d) V
```

---

### 5. Transformer Encoder Block

Each block contains:

```
LayerNorm
↓
Multi-Head Attention
↓
Residual Connection
↓
LayerNorm
↓
Feed Forward Network
↓
Residual Connection
```

---

## Training

The model is trained using a standard PyTorch training loop:

* Forward pass
* Loss computation
* Backpropagation
* Optimizer update

Example training output:

```
Epoch 1/10 | Loss: 1.82 | Train Acc: 45.23%
Epoch 2/10 | Loss: 1.21 | Train Acc: 61.10%
```

---

## Requirements

```
torch
torchvision
numpy
matplotlib
```

Install dependencies:

```
pip install torch torchvision
```

---

## Learning Goals

This project helped understand:

* How **transformers work on images**
* Implementing **multi-head attention manually**
* Understanding **tensor reshaping for attention**
* Role of **LayerNorm and residual connections**
* Vision Transformer architecture from first principles

---

## Future Improvements

* Add **Dropout**
* Add **Data Augmentation**
* Implement **training on larger datasets**
* Experiment with deeper transformer blocks
* Add **visualization of attention maps**

---

## References

* Vision Transformer (ViT) Paper
* Attention Is All You Need

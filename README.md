# Lipreading using Temporal Convolutional Networks

## Introduction

This repo contains the code for two lipreading model: DenseNet3D and model with Temporal Convolution Network. The first one is composed of 3D-convolutions. The second one contains 3D-convolution, ResNet and Multibranch TCN. This is an implementation of article [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). The author used pretrain weights from English speaking dataset LRW and then trained with his own Russian speaking dataset.

## How to test

To evaluate on LRW-structure dataset:

```Shell
!python main.py options_lip.toml 
```


# ResNet18-3DD
<div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px;">
    <img src="https://img.shields.io/github/license/AliKHaliliT/ResNet18-3DD" alt="License">
    <img src="https://github.com/AliKHaliliT/ResNet18-3DD/actions/workflows/tests.yml/badge.svg" alt="tests">
    <img src="https://img.shields.io/github/last-commit/AliKHaliliT/ResNet18-3DD" alt="Last Commit">
    <img src="https://img.shields.io/github/issues/AliKHaliliT/ResNet18-3DD" alt="Open Issues">
</div>
<br/>

A fully serializable 3D implementation of ResNet18, incorporating improvements from the paper ["Bag of Tricks for Image Classification with Convolutional Neural Networks"](https://arxiv.org/abs/1812.01187) along with additional personal optimizations and modifications.

This repository also includes implementations of the Hardswish and Mish activation functions:

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)

The codebase is fully integratable inside the TensorFlow and Keras code pipelines.

## Key Enhancements
- **Modified Stem:** Utilizes three convolutional layers instead of a single one.
- **ResNet-B Inspired Strides:** Moved the stride placement in the residual blocks from the first convolution to the second.
- **ResNet-D Inspired Shortcut:** Introduces an average pooling layer before the 1x1 convolution in the shortcut connection.
- **Reduced Downsampling:** Downsampling is now performed only twice (in the stem block) instead of the original five times.

<br/>
<br/>
<div align="center" style="display: flex; justify-content: center; align-items: center;">
    <img src="util_resources/readme/resnet_c.png" alt="ResNet-C image from the paper" style="width:300px; height:auto; margin-right: 16px;">
    <img src="util_resources/readme/shortcut.png" alt="Shortcut image by author" style="width:350px; height:auto;">
</div>
<br/>

*Note: The images above represent the architectural modifications. They depict 2D convolutional layers, whereas this project is focused on 1D convolutions. The ResNet-C image is sourced from the referenced paper, while the shortcut image is created by the author.*

## Installation & Usage
This code is compatible with **Python 3.12.8** and **TensorFlow 2.18.0**.

```python
from ResNet183DD import ResNet183DD


model = ResNet183DD(1000)
model.build((None, 32, 256, 256, 3))
model.summary()
```

### Model Summary Example
```bash
Model: "res_net183dd"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv3d_layer (Conv3DLayer)           │ (None, 16, 128, 128, 32)    │           2,592 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv3d_layer_1 (Conv3DLayer)         │ (None, 16, 128, 128, 32)    │          27,648 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv3d_layer_2 (Conv3DLayer)         │ (None, 16, 128, 128, 64)    │          55,296 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling3d (MaxPooling3D)         │ (None, 8, 64, 64, 64)       │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd (Residual3DD)            │ (None, 8, 64, 64, 64)       │         221,184 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_1 (Residual3DD)          │ (None, 8, 32, 32, 128)      │         671,744 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_2 (Residual3DD)          │ (None, 8, 32, 32, 128)      │         884,736 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_3 (Residual3DD)          │ (None, 8, 16, 32, 256)      │       2,686,976 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_4 (Residual3DD)          │ (None, 8, 16, 32, 256)      │       3,538,944 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_5 (Residual3DD)          │ (None, 8, 8, 16, 512)       │      10,747,904 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual3dd_6 (Residual3DD)          │ (None, 8, 8, 16, 512)       │      14,155,776 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling3d             │ (None, 512)                 │               0 │
│ (GlobalAveragePooling3D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         131,328 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 33,124,128 (126.36 MB)
 Trainable params: 33,124,128 (126.36 MB)
 Non-trainable params: 0 (0.00 B)
```

## License
This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.
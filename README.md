# EPro-PnP-6DoF (Modified Backbone Version)

<img src="viz.gif" width="550" alt=""/>

This repository contains a **modified** PyTorch implementation of End-to-End Probabilistic Perspective-n-Points (EPro-PnP) for 6DoF object pose estimation. It is based on the [official implementation](https://github.com/tjiiv-cprg/EPro-PnP) associated with the paper [[CVPR 2022] EPro-PnP](https://arxiv.org/pdf/2203.13254.pdf).

This project extends the original work by benchmarking various modern backbones to evaluate their effectiveness in high-precision pose estimation.

## Introduction & Modifications

### Original Architecture
EPro-PnP-6DoF reuses the off-the-shelf 6DoF pose estimation network CDPN. The original CDPN adopts two decoupled branches: a direct prediction branch for position, and a dense correspondence branch (PnP-based) for orientation. EPro-PnP-6DoF keeps only the dense correspondence branch (with minor modifications to the output layer for the 2-channel weight map), to which the EPro-PnP layer is appended for end-to-end 6DoF pose learning.

<img src="./architecture.png" width="450" alt=""/>

### Our Modifications: Alternative Backbones
In this project, we replaced the standard ResNet backbone with three state-of-the-art architectures to investigate the impact of **spatial resolution** and **global context** on pose accuracy, specifically focusing on strict metrics (e.g., 2cm/2deg).

We implemented the following backbones for comparison:

1. **Swin Transformer**:
    * A hierarchical Transformer using shifted windows for self-attention.
    * **Goal:** To leverage global context for better object structure understanding.

2. **ConvNeXt**:
    * A modernized CNN architecture that competes with Transformers in performance.
    * **Goal:** To serve as a balanced baseline between efficiency and accuracy.

3. **HRNet (High-Resolution Net)**:
    * Maintains high-resolution representations throughout the network.
    * Features a lightweight prediction head (0 hidden layers) compared to the original design.
    * **Goal:** To verify if high-resolution feature streams improve localization precision.

## Environment

The code has been tested in the following environment (Updated for RTX 50 Series / Modern GPUs):

- **OS**: WSL2 (Ubuntu 24.04)
- **Python**: 3.10
- **PyTorch**: 2.10.0.dev20251207+cu128
- **GPU**: NVIDIA RTX 5070ti

An example script for installing the python dependencies under CUDA 12.9:

```bash
# Create conda environment
conda create -y -n epropnp_6dof python=3.10
conda activate epropnp_6dof

# Install pytorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install opencv-python pyro-ppl PyYAML matplotlib termcolor plyfile easydict scipy progress numba tensorboardx
```

## Data Preparation

Please refer to [this link](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi#prepare-the-dataset) for instructions. Alternatively, you can download the complete dataset from [here](https://drive.google.com/drive/folders/1o53iYMvY5S8Tc1SxvQThWJJfFRH6ZV5w?usp=sharing). Afterwards, the dataset folders should be structured as follows:

```
EPro-PnP-6DoF/
├── dataset/
│   ├── bg_images/
│   │   └── VOC2012/
│   └── lm/
│       ├── models/
│       │   ├── ape/
│       │   …
│       ├── imgn/
│       │   ├── ape/
│       │   …
│       ├── real_test/
│       │   ├── ape/
│       │   …
│       └── real_train/
│           ├── ape/
│           …
├── lib/    
├── tools/
…
```

## Models & Benchmark Results

We evaluated four different backbone architectures on the **LineMOD** dataset (13 objects, standard split) for 160 epochs. The results demonstrate the performance trade-offs between different architectures, particularly in high-precision scenarios.

### Detailed Evaluation Metrics

The table below reports the **ADD(-S)** metrics and **n° n cm** metrics (Spatial Accuracy) at various thresholds.

| Backbone Model | Config | Dataset | ADD 0.02d | ADD 0.05d | ADD 0.10d | **ADD Mean** | 2°, 2 cm | 5°, 5 cm | 10°, 10 cm | **Spc Mean** | Download |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet** | [basic](tools/exps_cfg/epropnp_basic.yaml) | LineMOD | 32.99 | 73.28 | 92.59 | 61.83 | 65.79 | 96.72 | 99.57 | 85.88 | [Link](https://drive.google.com/file/d/1p9OVl2f0rD9JPP_kYthgIIT_hE5V4FcV/view?usp=sharing) |
| **Swin Transformer** | [swin](tools/exps_cfg/epropnp_swin_basic.yaml) | LineMOD | 36.13 | 76.39 | **94.55** | 64.47 | 67.42 | 97.46 | 99.75 | 86.63 | [Link](https://drive.google.com/file/d/1qpcYQPokkHx-bkNDstrqDBWJXVTEDb-g/view?usp=drive_link) |
| **ConvNeXt** | [convnext](tools/exps_cfg/epropnp_convnext_basic.yaml) | LineMOD | 38.28 | 77.11 | 94.27 | 65.26 | 71.02 | **97.73** | **99.84** | 87.68 | [Link](https://drive.google.com/file/d/1m2ffe2_J-CzL0RzD-kPGdV7LXKOaZq3k/view?usp=sharing) |
| **HRNet** | [hrnet](tools/exps_cfg/epropnp_hrnet_basic.yaml) | LineMOD | **41.21** | **78.58** | 93.98 | **66.63** | **76.53** | 97.10 | 98.92 | **88.64** | [Link](https://drive.google.com/file/d/1eoxM3AH6cr-PLLlDHBwfFZ9wBbnVPgdO/view?usp=drive_link) |

> **Key Findings:**
> * **High Precision:** HRNet significantly outperforms other models in strict metrics (**ADD 0.02d** and **2°, 2 cm**), showing superior capability in precise localization.
> * **Global Context:** Swin Transformer achieves the best performance in the loose metric (**ADD 0.10d**), indicating strong object recognition capabilities.
> * **Balance:** ConvNeXt offers a balanced performance, achieving the highest accuracy in the practical **5°, 5 cm** threshold.
## Train

To start training, enter the directory `EPro-PnP-6DoF`, and run:

### ResNet
```bash
python tools/main.py --cfg tools/exps_cfg/epropnp_basic.yaml
```

### HRNet
```bash
python tools/main.py --cfg tools/exps_cfg/epropnp_hrnet_basic.yaml
```

### Swin
```bash
python tools/main.py --cfg tools/exps_cfg/epropnp_swin_basic.yaml
```

### ConvNeXt
```bash
python tools/main.py --cfg tools/exps_cfg/epropnp_convnext_basic.yaml
```

By default GPU 0 is used, you can set the environment variable `CUDA_VISIBLE_DEVICES` to change this behavior.
Checkpoints, logs and visualizations will be saved to `EPro-PnP-6DoF/exp`.

## Test

To evaluate the trained models on the LineMOD test split:

1.  Download the pre-trained weights (`model.pth`) from the table above.
2.  Open the corresponding config file (e.g., `tools/exps_cfg/epropnp_hrnet_basic.yaml`).
3.  Set `load_model` to the path of your downloaded file (e.g., `'path/to/epropnp_hrnet_basic.pth'`).
4.  Change `test` from `False` to `True`.

```bash
python tools/main.py --cfg tools/exps_cfg/{corresponding config file name}.yaml
```

Logs and visualizations will be saved to `EPro-PnP-6DoF/exp`.
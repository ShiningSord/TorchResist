# Table of Contents

- [Introduction to TorchResist](#introduction-to-torchresist)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Prepare Mask](#prepare-mask)
    - [7nm Resolution Masks](#7nm-resolution-masks)
  - [Litho Simulation](#litho-simulation)
    - [ICCAD13](#iccad13)
    - [FuILT](#fuilt)
    <!-- - [Potential Third Option](#potential-third-option) -->
  - [Resist Simulation](#resist-simulation)
    - [Usage](#usage)
    - [Features](#features)

- [Extension: Advanced Usage Techniques](#extension-advanced-usage-techniques)

---

# Introduction to TorchResist
A Neat Litho-Resist Simulator

The litho model part is forked from FuILT. We calibrate the model parameters on real designs.

We implemented the resist model from scratch and refined it with a calibration set.

---

# Installation

Follow these steps to configure an AnaConda environment named `torchresist`:

**Step 1:** Create a new anaconda environment called torchresist and activate it using following commands:

```bash
conda create -n torchresist python==3.9
conda activate torchresist
```

**Step 2:** To install the required dependencies for this project, ensure that the requirements.txt file is in the current directory, then run the following command:

```bash
pip3 install -r requirements.txt
```

---

# Quick Start

First, clone the repository:

```bash
git clone https://github.com/your-repo-url.git
```

## Prepare Mask

Refer to [LithoBench’s GitHub](https://github.com/shelljane/lithobench) for downloading mask data and organize it into the required structure:

1. Create a `data` folder in the root directory.
2. Inside `data`, create subdirectories for different mask sources, e.g., `Dataset1`, `Dataset2`.
3. Within each dataset, create a `mask` folder containing `Images` and `Numpys` subfolders:
   - `Images`: Store binary mask images named as `mask000000.png` (six-digit format, starting from 0).
   - `Numpys`: Store mask arrays in NumPy `bool` format with shape `[B, H, W]`.

Provide a script to automate this process:

```bash
bash scripts/processmask.sh path/to/download.zip
```

The final structure:

```
data/dataset1/mask/1nm/images/mask000000.png
data/dataset1/mask/1nm/numpys/mask.npy
```

Demo mask images are stored in `demo/mask/`.

### 7nm Resolution Masks

To enhance efficiency, masks can be downsampled to 7nm resolution using a script:

```bash
python3 tools/downsampling.py --input path/to/1nm/mask --output path/to/7nm/mask
```

Output structure:

```
data/dataset1/mask/7nm/images/mask000000.png
data/dataset1/mask/7nm/numpys/mask.npy
```

---

## Litho Simulation

We provide two Litho Model options: ICCAD13 and FuILT.

### ICCAD13

1. Two benchmarks are referred to:
   
- S. Banerjee, Z. Li, and S. R. Nassif, “ICCAD-2013 CAD contest in mask optimization and benchmark suite,” in IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2013, pp. 271–274.

- S. Zheng etc. *lithobench*. Github, 2023, https://github.com/shelljane/lithobench.

2. Note: Masks used here have a fixed resolution of 1nm.
3. Use the script to generate lithography results:

```bash
python3 tools/litho_iccad13.py --mask path/to/1nm/mask/numpy.npy --outpath path/to/output
```

Output structure:

```
data/dataset1/litho/iccad13/images/litho000000.png
data/dataset1/litho/iccad13/numpys/litho.npy
```

### FuILT

1. Use masks with optional 1nm resolution (controlled via a parameter).
2. Generate lithography results:

```
python3 -m examples.fuilt \
  --mask ./data/Dataset1/1nm/images \
  --resolution 1.0 \
  --outpath ./data/Dataset1/fuilt/1nm/litho
```

Output structure:

```
path/to/output/images/cell000000.png
path/to/output/numpys/cell000000.npy
```

<!-- ### Potential Third Option

**Reserved for future updates.**

Demo results are stored in `demo/litho/ICCAD13/` and `demo/litho/FuILT/`. -->

---

## Resist Simulation

TorchResist provides resist parameters for various optical lithography solutions.

### Usage

Simulate resist with the provided script:

```
python3 -m examples.resist \
  --lithomodel FUILT \
  --lithoresult ./data/Dataset1/fuilt/1nm/litho/numpys \
  --outpath ./data/Dataset1/fuilt/1nm/resist \
  --resolution 1.0
```

- `--lithomodel`: Choose `ICCAD13` or `FUILT`.
- `--lithoresult`: Path to the `.npy` lithography result.
- `--outpath`: Directory for output files.
- `--resolution`: Input resolution in nm (default: `1.0`).

### Features

- **Customizable Parameters:** Adjust resist settings via input arguments for different lithography models and resolutions.
- **Flexible Resolution:** By default, the tool assumes a resolution of 1nm. While this resolution has not been rigorously validated, you can modify it based on your requirements without significantly impacting the results.



---


# Extension: Advanced Usage Techniques

Reserved for advanced instructions and optimizations.

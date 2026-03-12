# GK-SMOTE

[![Read Article](https://img.shields.io/badge/Springer-Full%20Article-blue?logo=springer)](https://link.springer.com/chapter/10.1007/978-981-95-5719-6_13)

A research-oriented Python implementation of **GK-SMOTE**, a Gaussian-kernel-density-guided oversampling method for **imbalanced and noisy binary classification**.

## Overview

Class imbalance is a common challenge in machine learning, where the minority class is underrepresented and often harder to learn accurately. Traditional oversampling methods such as **SMOTE** can improve minority representation, but they may also generate low-quality synthetic samples in noisy or overlapping regions.

**GK-SMOTE** addresses this issue by introducing a **Gaussian kernel density-guided oversampling strategy**. The method first filters potentially noisy minority samples, estimates local density among minority instances, separates them into safer and more borderline regions, and then generates synthetic samples more adaptively.

This repository provides a clean and extensible Python implementation inspired by the GK-SMOTE paper for experimentation, benchmarking, and future research.


## Key Idea

GK-SMOTE improves oversampling by combining:

- **noise-aware minority filtering**
- **Gaussian kernel density estimation**
- **safe vs borderline minority region separation**
- **adaptive synthetic sample generation**

Compared with standard SMOTE, GK-SMOTE is designed to be more robust in the presence of **label noise** and **class overlap**.


## Features

- Python implementation of GK-SMOTE
- Designed for **binary imbalanced classification**
- Noise-resilient synthetic sample generation
- Modular code structure for research and extension
- Easy integration with scikit-learn workflows
- Example scripts and test cases included


## Repository Structure

```text
GK-SMOTE/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── gksmote/
│   ├── __init__.py
│   ├── gksmote.py
│   ├── utils.py
│   └── metrics.py
│
├── examples/
│   ├── demo_synthetic.py
│   └── compare_with_smote.py
│
├── notebooks/
│   └── gk_smote_demo.ipynb
│
├── tests/
│   ├── test_gksmote.py
│   └── test_utils.py
│
└── data/
    └── .gitkeep
```

## Installation
Clone the repository and install dependencies:
```
git clone https://github.com/Miraj-Rahman-AI/GK-SMOTE.git
cd GK-SMOTE
pip install -r requirements.txt
pip install -e .
```

## How GK-SMOTE Works

The implementation follows the core idea of the paper:

1. Identify minority samples in the dataset
2. Detect potentially noisy minority samples using neighborhood structure
3. Estimate local minority density using Gaussian kernel-based ideas
4. Split minority samples into safe and borderline groups
5. Generate synthetic samples through interpolation within minority neighborhoods
6. Return the resampled dataset for downstream classification


## Current Limitations

- Currently supports **binary classification only**
- This is a **research-oriented implementation**, not an official release from the original authors
- Some engineering details may differ from the original experimental environment in the paper
- More benchmarking on real-world datasets can be added


## Future Work

Planned improvements include:

- multiclass extension
- more faithful reproduction of the paper’s experimental protocol
- integration with imbalanced-learn style API
- dataset benchmarking pipeline
- visualization tools for synthetic sample generation
- GitHub Actions for automated testing


## Citation

If you use this repository in your research, please cite the original paper.

You may also cite this implementation as:

```bibtex
@inproceedings{miraj2025gksmote,
  author    = {Miraj, M. R. and Huang, H. and Yang, T. and Zhao, J. and Mu, N. and Lei, X.},
  title     = {GK-SMOTE: A Hyperparameter-Free Noise-Resilient Gaussian KDE-Based Oversampling Approach},
  booktitle = {Asia-Pacific Web (APWeb) and Web-Age Information Management (WAIM) Joint International Conference on Web and Big Data},
  pages     = {197--212},
  year      = {2025},
  month     = aug,
  address   = {Singapore},
  publisher = {Springer Nature Singapore},
  url       = {https://link.springer.com/chapter/10.1007/978-981-95-5719-6_13},
  doi       = {10.1007/978-981-95-5719-6_13}
}
```


##  Author
**[Miraj Rahman](https://github.com/Miraj-Rahman-AI)**  
AI Researcher | Autonomous Agents | RAG Systems | Trustworthy AI

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is released under the [MIT License](LICENSE).


##  Support
If this project supports your research or learning,
please consider giving it a ⭐ on GitHub.

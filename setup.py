from setuptools import setup, find_packages

setup(
    name="gksmote",
    version="0.1.0",
    description="Python implementation of GK-SMOTE for imbalanced and noisy binary classification",
    author="Mahabubur Rahman Miraj",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "imbalanced-learn>=0.11",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)

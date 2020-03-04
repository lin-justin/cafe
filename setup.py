import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = "cafe",
    version = "2.0.5",
    author = "Justin Lin",
    description = 'Classifying Antibodies for Expression',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lin-justin/cafe",
    packages = setuptools.find_packages(),
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires = [
        "biopython>=1.73",
        "matplotlib>=3.1.2",
        "numpy>=1.17.4",
        "pandas>=0.24.2",
        "scikit-learn>=0.22",
        "scikit-plot==0.3.7",
        "seaborn==0.9.0",
        "imbalanced-learn==0.6.1",
        "eli5==0.10.1"
    ],
    python_requires = ">=3.7",
)
# cafe: Classifying Antibodies for Expression

## About

A Python package, or wrapper, that transforms FASTA files into a data format suitable for machine learning classification.

This is a research project during my computational biology internship at EMD Serono, Research and Development Institute in Billerica, MA under [Dr. Yves Fomekong Nanfack](yves.fomekong.nanfack@emdserono.com). The idea is to have each amino acid as a feature (with its respective numerical value based on solubility, hydrophobicity, etc. from the literature) and see if it is possible to predict antibodies that have high affinity or other ideal characteristics.

The `data_transform` module reads in FASTA files, extracts the amino acid sequence, splits each amino acid into its own columns, replaces each amino acid with a value provided by the user (as a file), and outputs a pandas dataframe(s).

The `ml` module performs classic machine learning tasks such as splitting the data, standardizing the data, model selection, training, and evaluating.


## Installation

**Please have Python 3.7 installed.**

Once Python 3.7 is installed, you can install the package as so:

`pip3 install cafe`

## Usage

Please see the examples folder.# cafe

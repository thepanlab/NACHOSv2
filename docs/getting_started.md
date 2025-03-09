# Getting Started with Image Classification ML Pipeline

Welcome to our image classification machine learning pipeline. This guide will help you set up and prepare to use the pipeline on a supercomputer environment.

## Operating System

NACHOSv2 was tested on Linux (Ubuntu 20.04)

## Installation

We suggest that you install NACHOSv2 in a virtual environment. We would show to steps for creating a conda environment.

```bash
conda create --name nachosv2
conda activate nachosv2
```

1. Install PyTorch from [this link](https://pytorch.org/get-started/locally/)


2. Install
```bash
   git clone https://github.com/thepanlab/NACHOS_v2.git
   cd NACHOS_v2
   pip install -e .
```

Using `-e`, makes an editable installation, that is, the modifications in the library will be taken into account. 

## Configuration

The pipeline uses a configuration file in YAML for setting up training parameters. You'll need to modify this file before running a training session.

Detailed instructions on configuring the pipeline will be provided in the [Using the Pipeline](using_the_pipeline.md) section.

# Getting Started with Image Classification ML Pipeline

Welcome to our image classification machine learning pipeline. This guide will help you set up and prepare to use the pipeline on a supercomputer environment.

## Prerequisites

Before you begin, ensure you have the following:

1. **Access to a Supercomputer**:
   - This pipeline is designed to run on high-performance computing environments.
   - Ensure you have the necessary permissions and access to the supercomputer.

2. **Conda**:
   - We use Conda for managing the Python environment and dependencies.
   - Ensure Conda is installed and accessible from your command line.

3. **Git**:
   - For cloning the repository from GitHub.

Note: Exact hardware requirements (RAM, GPU) may vary depending on the specific model and image dataset you'll be working with.

## Installation

Follow these steps to set up the pipeline:

1. **Clone the repository**:
   ```
   git clone https://github.com/thepanlab/pipeline_pytorch.git
   cd pytorch-pipeline
   ```

2. **Create and activate the Conda environment**:
   We provide a YAML file that specifies all necessary dependencies. Use it to create your environment:
   ```
   conda env create -f environment.yml
   conda activate pytorch_gpu_env  # The name might differ; use the name specified in environment.yml
   ```

   This step is crucial when working on a shared supercomputer to avoid conflicts with other users' environments.

3. **Verify installation**:
   After activating the environment, you can verify that all required packages are correctly installed by running:
   ```
   conda list
   ```
   This will show all installed packages in your environment.

## Configuration

The pipeline uses a configuration file for setting up training parameters. You'll need to modify this file before running a training session. Detailed instructions on configuring the pipeline will be provided in the [Using the Pipeline](using_the_pipeline.md) section.

## Next Steps

Now that you have the pipeline installed, proceed to the [Pipeline Overview](pipeline_overview.md) to understand its components, or go to [Using the Pipeline](using_the_pipeline.md) for instructions on how to run it, including how to set up the configuration file for training.

For basic usage and commands, refer to the [Basic Usage](basic_usage.md) guide.

## Test Dataset

We have included a convenient script to download and set up the CIFAR-10 dataset for testing purposes. To use it:

1. Ensure you are in the root directory of the project.

2. Run the following command:
    ```
    python3 src/setup/create_cifar10_dataset.py 
    ```
    This script will download CIFAR-10, extract it, organize it in a format suitable for our pipeline and creates the csv files and th configuration file.

3. Once the script completes, you'll have a ready-to-use test  dataset in the data directory.

You can use this dataset to verify that your pipeline is working correctly or to experiment with different configurations. For instructions on how to use this dataset with the pipeline, refer to the [Using the Pipeline](using_the_pipeline.md) section.

Note: While CIFAR-10 is excellent for testing, remember that it may not be representative of your actual use case. We recommend using it primarily for initial setup and testing before moving on to your specific datasets.

## Important Notes

1. Always ensure you're in the root directory of the project when running commands related to this pipeline.

2. Always activate the Conda environment before running any pipeline commands:
   ```
   conda activate image_classification_env  # Use the correct environment name
   ```

If you encounter any issues during installation or setup, please refer to our [Troubleshooting](troubleshooting.md) guide.
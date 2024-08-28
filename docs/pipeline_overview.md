# Pipeline Overview

This document provides an overview of our image classification pipeline, designed to run on supercomputer environments. The pipeline is flexible, supporting various types of image classification models and utilizing nested cross-validation for robust evaluation.

## Pipeline Flow

1. **Command Line Analysis**:
   - The pipeline begins by parsing command line arguments to gather necessary information for the run.

2. **Data Normalization**:
   - For 2D images: The pipeline normalizes the data based on input statistics.
   - For 3D images: No normalization is applied.

3. **Model Training**:
   - The core of the pipeline, where model training occurs using nested cross-validation.

4. **Results Processing**:
   - Post-training, users can employ additional commands to process and analyze results.

## Key Components

### 1. Data Handling
- Data is read from CSV files and stored in a dictionary structure.
- For each fold in the cross-validation, separate datasets are created.

### 2. Model Support
The pipeline is designed to be model-agnostic, supporting various architectures:
- Built-in test model for CIFAR-10
- 2D InceptionV3 model
- Custom 3D model

Users can also implement and use their own models within the pipeline.

### 3. Training Process
The training process follows a nested structure:
- `training_inner_loop.py`: Initiates the training process
  - Utilizes `sequential_processing.py`
    - Calls `sequential_subject_loop.py` for each subject
      - Employs `training_loop.py`
        - Uses `training_fold.py` for each rotation in the nested cross-validation

### 4. Cross-Validation
- The pipeline implements nested cross-validation for robust model evaluation.
- Different phases (training, validation, testing) are handled within `training_fold.py`.
- A loop over phases executes specific operations based on the current phase.

## Usage Flow

1. **Setup**: Users start by setting up the environment and data as described in the Getting Started guide.

2. **Configuration**: Adjust any necessary parameters or configurations.

3. **Training**: Launch the training process using the appropriate command-line interface.

4. **Evaluation**: After training, use provided tools to process and analyze results.

## Notes on Parallelization

Currently, the pipeline operates sequentially. Future versions may implement parallelization to leverage supercomputer capabilities more effectively.

## Extensibility

The pipeline's modular design allows for easy extension:
- New models can be added by implementing them and integrating them into the training loop.
- Additional data preprocessing steps can be incorporated as needed.
- Result processing tools can be expanded to provide more in-depth analysis.

For detailed information on using specific components of the pipeline, please refer to the respective documentation sections.
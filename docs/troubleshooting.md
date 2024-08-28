# Troubleshooting Guide

This guide addresses common issues you might encounter while using the image classification pipeline and provides solutions to resolve them.

## Installation and Environment Issues

### Problem: Conda environment creation fails
1. Ensure you have the latest version of Conda installed.
2. Check if you have sufficient disk space.
3. Try creating the environment with `conda create -n myenv python=3.x` and then install packages manually.

### Problem: Missing dependencies
1. Activate your Conda environment: `conda activate myenv`
2. Install missing packages: `conda install package_name`

## Data Preparation Issues

### Problem: CSV files not found
1. Verify that your CSV files are in the correct directory (`data/your_data_csv/`).
2. Check file permissions.
3. Ensure file names match those specified in your configuration file.

### Problem: Image files not loading
1. Confirm that the image file path in your CSV is correct.
2. Check if image files exist in the specified location.
3. Verify image file formats are supported (e.g., .jpg, .png).

## Training Issues

### Problem: Out of memory error
1. Reduce the batch size in your configuration file.
2. If using a 3D model, consider reducing image dimensions or using a 2D model if possible.
3. Check if other processes are consuming GPU memory and close them if necessary.

### Problem: Training not starting
1. Ensure you're in the correct directory when running the training command.
2. Verify that all paths in your configuration file are correct.
3. Check for syntax errors in your configuration JSON file.

### Problem: NaN loss values
1. Check your learning rate - it might be too high. Try reducing it in the configuration file.
2. Verify that your input data is normalized correctly.
3. Check for any division by zero in custom loss functions or metrics.

## Results Processing Issues

### Problem: Processing scripts not running
1. Ensure you're using the correct Python environment where dependencies are installed.
2. Check that you're in the correct directory when running the script.
3. Verify that the input paths in your processing configuration file are correct.

### Problem: Graphs not generating
1. Check if you have write permissions in the output directory.
2. Ensure you have all required plotting libraries installed (e.g., matplotlib).
3. Verify that the data files exist and are not empty.

## Supercomputer-Specific Issues

### Problem: Job submission fails
1. Check your job submission script for errors.
2. Verify that you have the necessary permissions and allocation on the supercomputer.
3. Ensure you're not exceeding resource limits (CPU, GPU, memory, time).

### Problem: Job terminates unexpectedly
1. Check the error logs provided by the job scheduler.
2. Verify that your job doesn't exceed the requested resources.
3. Ensure your script handles exceptions properly to provide meaningful error messages.

## General Troubleshooting Tips

1. Always check the console output and error messages for clues about what went wrong.
2. Verify that you're using the most recent version of the pipeline and all its dependencies.
3. If a problem persists, try running your script with verbose logging enabled.
4. For complex issues, consider sharing relevant parts of your configuration file and error logs (make sure to remove any sensitive information).

If you encounter an issue not covered in this guide, or if the provided solutions don't resolve your problem, please reach out to the support team or open an issue in the project's repository.
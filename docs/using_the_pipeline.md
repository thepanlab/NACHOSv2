# Using the Pipeline

This guide provides detailed instructions on how to use our image classification pipeline, from setup to result processing.

## Preparation

Before running the pipeline, ensure you have:

1. Activated your Conda environment:
   ```
   conda activate pytorch_gpu_env
   ```

2. Prepared your CSV files in the `data/your_data_csv/` directory.

3. Modified a configuration file in `scripts/config_files/` with the necessary training parameters and informations.

## Running the Pipeline

### Basic Command

To start training, use the following command:

```bash
python3 -m scripts.training.training_sequential.training_inner_loop --file scripts/config_files/3D_config_inner_conv3D_trial_parallel.json
```

Replace the JSON file name with your specific configuration file.

### Important Arguments

- `--file` or `--config_file`: Specify a single configuration file.
- `--folder` or `--config_folder`: Specify a folder containing multiple configuration files to run several training sessions.
- `--verbose` or `--v`: Activate verbose mode for more detailed output.
- `--device` or `--d`: Choose the CUDA device for execution. Default is cuda:1.

### Example

```bash
python3 -m scripts.training.training_sequential.training_inner_loop --file scripts/config_files/my_custom_config.json --verbose --device cuda:0
```

## Monitoring Progress

Training progress information will be displayed in the console. Keep an eye on this output to monitor the advancement of your training session.

## Using Screen for Long Training Sessions

For long training sessions, it's recommended to use the `screen` command. This allows you to detach from the session without interrupting the training, even if your connection is lost.

1. Start a new screen session:
   ```
   screen -S training_session
   ```

2. Run your training command in this session.

3. Detach from the session: Press `Ctrl+A`, then `D`.

4. To reattach later:
   ```
   screen -r training_session
   ```

## Processing Results

After training, use the following command structure to process results:

```bash
python3 src/results_processing/[processing_type]/[specific_script].py
```

Example:
```bash
python3 src/results_processing/learning_curve/learning_curve_many.py
```

Note: Each result processing function requires its own configuration file.

## Customizing the Pipeline

### Adding Custom Models

1. Create a new file in `src/model_processing/models/` for your model.
2. Add the model name to the dictionary of possible models in `src/model_processing/create_model.py`.

## Best Practices and Tips

1. Always double-check your configuration file before starting a training session.
2. Use verbose mode (`--verbose`) for detailed logs during training.
3. When running on a shared system, be mindful of resource usage and use the `--device` argument to specify which GPU to use.
4. Regularly backup your configuration files and results.
5. For reproducibility, consider versioning your configuration files alongside your code.

## Troubleshooting

If you encounter issues:

1. Check the console output for error messages.
2. Verify that all required data files are in the correct locations.
3. Ensure your configuration file is correctly formatted and contains all necessary parameters.
4. Check that you're using the correct Python environment with all required dependencies installed.

For persistent issues, refer to the troubleshooting section in the documentation or contact the support team.
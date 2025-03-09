# Using the Pipeline

1. The metadata (label, fold, etc) shoould be located in a CSV file.

The CVS should have minimum the colums: `fold_name`,`absolute_filepath`,`label`. e.g.

| fold_name | absolute_filepath | label |
|-----------|-------------------|-------|
|k1|/home/pcallec/NACHOS_v2/data/pig_kidney_subset/k1/k1_cortex/100_k1_cortex.jpg|0|
|k1|/home/pcallec/NACHOS_v2/data/pig_kidney_subset/k1/k1_cortex/10_k1_cortex.jpg|0|
|k1|/home/pcallec/NACHOS_v2/data/pig_kidney_subset/k1/k1_cortex/11_k1_cortex.jpg|0|
|...|...|...|

`fold_name`: data will be split according to this column \
`absolute_filepath`: absolute path for the file \
`label`: category in integer values e.g. `0`,`1`,...

The CSV can have more columns; however, they won't be used.

## Running the Pipeline

### Basic Command

To start training, use the following command:

```bash
python3 NACHOSv2_train --loop "cross-validation" --file config_inner_conv3D_trial_parallel.yml --device cuda:0
```

- `--file` or `--config_file`: Specify a single configuration filepath.
- `--loop`: selects between `cross-validation` or `cross-testing`
- `--folder` or `--config_folder`: Specify a folder containing multiple configuration files to run several training sessions.
- `--verbose` or `--v`: Activate verbose mode for more detailed output.
- `--device` or `--d`: Choose the CUDA device for execution. default `cuda:0` 

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
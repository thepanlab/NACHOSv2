# Using the Pipeline

1. The metadata (label, fold, etc) shoould be located in a CSV file.

The CVS should have minimum the colums: `fold_name`,`absolute_filepath`,`label` (0-indexed). e.g.

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

#### Training

To start training, use the following command:

```bash
python NACHOSv2_train --loop "cross-validation" --device cuda:0 --file config_training.yml 
```

- `--file` or `--config_file`: Specify a single configuration filepath.
- `--loop`: selects between `cross-validation` or `cross-testing`
- `--folder` or `--config_folder`: Specify a folder containing multiple configuration files to run several training sessions.
- `--verbose` or `--v`: Activate verbose mode for more detailed output.
- `--device` or `--d`: Choose the CUDA device for execution. default `cuda:0` 

#### Get summary

To get summary results:
```bash
python NACHOSv2_get_summary --file config_summary_cv.yml
```
If are getting summary for `cross-validation`, it will automatically generate the configuration files to train `for cross-testing` inside the folder ``

#### Get confusion matrix

To get confusion matrix:
```bash
python3 NACHOSv2_get_confusion_matrix --file config_confusionmatrix_cv.yml
```

### Get learning curve

To get learning curve:
```bash
python3 NACHOSv2_get_learning_curve --file config_confusionmatrix_cv.yml
```

### Get predictions

```bash
python3 NACHOSv2_get_predictions --file config_confusionmatrix_cv.yml
```

## Troubleshooting

If you encounter issues:

1. Check the console output for error messages.
2. Verify that all required data files are in the correct locations.
3. Ensure your configuration file is correctly formatted and contains all necessary parameters.
4. Check that you're using the correct Python environment with all required dependencies installed.

For persistent issues, refer to the troubleshooting section in the documentation or contact the support team.
# Results Processing Guide

This guide provides information on how to process and interpret the results from your image classification pipeline training.

## Overview

After training, the pipeline doesn't automatically generate processed results. You have access to raw accuracy and loss data. To gain deeper insights, you need to use specific scripts for result processing.

## Available Processing Scripts

The pipeline offers several scripts for processing results, located in the `results_processing` directory:

- `learning_curve`: Plots accuracy and validation curves over epochs.
- `class_recall`: Calculates recall for each class at each epoch.
- `confusion_matrix`: Creates confusion matrices for model predictions.
- `epoch_counting`
- `grad_cam`
- `metrics_per_category`
- `metrics_table`
- `prediction`
- `random_search_post`
- `roc_curve`
- `summary_random_search`
- `summary_table`
- `tabled_prediction_info`

Note: Currently, `learning_curve`, `class_recall`, and `confusion_matrix` have been updated and are ready to use.

## Using Processing Scripts

To use a processing script, navigate to its directory and run it with Python. You can now specify the configuration file directly in the command line using the `--f` argument. For example:

```bash
python3 src/results_processing/learning_curve/learning_curve_many.py --f path/to/your/config.json
```

You can also use `--file` or `--config_file` as alternative flags to specify the configuration file.

If you don't specify a configuration file in the command line, the script will prompt you to enter the path to a configuration file. If you don't provide a file at this prompt, the script will use the default configuration file.

### Configuration Files

Each processing function has its own configuration file. Here's an example for the `learning_curve` script:

```json
{
  "input_path": "results/distributed/pig_kidney_subset_parallel/training_results/Test_subject_k1/config_pig_kidney_subset_InceptionV3/InceptionV3_test_k1_val_k2",
  "output_path": "results/processed_output/learning_curve",

  "training_loss_line_color": "b",
  "validation_loss_line_color": "r",
  "training_accuracy_line_color": "b",
  "validation_accuracy_line_color": "r",

  "font_family": "DejaVu Sans",
  "label_font_size": 12,
  "title_font_size": 12,

  "save_resolution": 600,
  "save_format": "png"
}
```

Modify these configuration files to customize the input/output paths, visual styles, and other parameters specific to each processing script.

## Output Formats

The processing scripts generate outputs in two main formats:
- Images (PNG): For visualizations like learning curves and confusion matrices.
- CSV files: For tabular data and metrics.

## Best Practices

1. **Save Processed Results**: Always save your processed results before running new training sessions or processing scripts. This prevents accidental loss of valuable insights.

2. **Consistent Naming**: Use consistent naming conventions for your output files to easily track and compare results from different runs.

3. **Regular Backups**: Periodically backup your results, especially for long-running experiments or important findings.

4. **Version Control**: Consider using version control for your configuration files to track changes in your processing parameters.

5. **Documentation**: Keep notes on the specific processing steps and parameters used for each set of results. This aids in reproducibility and interpretation.

## Interpreting Results

When interpreting your results:

1. **Learning Curves**: 
   - Look for convergence in both training and validation accuracy.
   - Watch for signs of overfitting (high training accuracy but low validation accuracy).

2. **Confusion Matrix**: 
   - Identify which classes are most often confused with each other.
   - Look for patterns that might indicate biases in your model or dataset.

3. **Class Recall**: 
   - Understand which classes are easier or harder for your model to identify correctly.
   - Consider if some classes might need more training data or attention.

## Future Developments

While currently there are no built-in tools for comparing results across different training runs, you can manually compare outputs from different processing runs. Consider developing custom scripts for this purpose if it becomes a frequent need.

Remember, the interpretation of results is often specific to your particular use case and dataset. Always consider the context of your project when drawing conclusions from these processed results.
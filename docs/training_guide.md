# Training Guide

This guide provides detailed information on training models using our image classification pipeline. It covers configuration, parameters, and best practices for effective model training.

## Configuration File

The training process is controlled by a JSON configuration file. Here's an example of its structure with explanations:

```json
{
    "hyperparameters": {
        "batch_size": 32,
        "channels": 1,
        "cropping_position": [40, 10],
        "decay": 0.001,
        "do_cropping": false,
        "epochs": 20,
        "learning_rate": 0.0005,
        "momentum": 0.9,
        "bool_nesterov": true,
        "patience": 10
    },

    "data_input_directory": "/ssd/sly/CW_treated_tiff",
    "csv_input_directory": "data/3D_kidney_csv",
    "output_path": "results/distributed/3D_pig_kidney_subset_parallel",
    "job_name": "3D_pig_kidney_subset",
    
    "k_epoch_checkpoint_frequency": 1,

    "shuffle_the_images": true,
    "shuffle_the_folds": false,
    "seed": 9,

    "class_names": ["cortex", "medulla", "pelvis-calyx"],
    "selected_model_name": "Conv3DModel",
    "metrics": [],
    
    "subject_list": ["Kidney_01", "Kidney_02", "Kidney_03", "Kidney_04", "Kidney_05", "Kidney_06", "Kidney_07", "Kidney_08", "Kidney_09", "Kidney_10"],
    "test_subjects": ["Kidney_01", "Kidney_02", "Kidney_03", "Kidney_04", "Kidney_05", "Kidney_06", "Kidney_07", "Kidney_08", "Kidney_09", "Kidney_10"],
    "validation_subjects": ["Kidney_01", "Kidney_02", "Kidney_03", "Kidney_04", "Kidney_05", "Kidney_06", "Kidney_07", "Kidney_08", "Kidney_09", "Kidney_10"],
    
    "image_size": [185, 210, 185],
    "target_height": 301,
    "target_width": 235
}
```

### Key Configuration Parameters

1. **Hyperparameters**:
   - `batch_size`: Number of samples per batch during training. Decrease this value if you encounter "out of memory" errors.
   - `channels`: Number of input channels in the image. Use 1 for grayscale images, 3 for RGB images.
   - `epochs`: Total number of training epochs.
   - `learning_rate`: Step size at each iteration while moving toward a minimum of the loss function.
   - `momentum`: Accelerates gradient descent in the relevant direction.
   - `bool_nesterov`: Whether to use Nesterov momentum.
   - `patience`: Number of epochs with no improvement after which training will be stopped.

2. **Data and Output**:
   - `data_input_directory`: Path to the input image data.
   - `csv_input_directory`: Path to CSV files containing data information.
   - `output_path`: Where to save the results.
   - `job_name`: Name given to the training job. This will be used for naming folders and files related to this training run.

3. **Training Process**:
   - `k_epoch_checkpoint_frequency`: How often to save model checkpoints.
   - `shuffle_the_images`: Whether to shuffle images during training.
   - `shuffle_the_folds`: Whether to shuffle the folds in cross-validation.
   - `seed`: Random seed for reproducibility.

4. **Model and Data Specifics**:
   - `class_names`: List of class names for classification.
   - `selected_model_name`: The model architecture to use. This must be a key in the model dictionary in `src/model_processing/model_creator.py`.
   - `metrics`: List of metrics to track during training. By default, it includes accuracy and loss. You can also add 'recall' if needed.
   - `subject_list`: Subjects to use for training.
   - `test_subjects`: Subjects to use for testing.
   - `validation_subjects`: Subjects to use for validation.
   - `image_size`: Dimensions of the input images.
   - `target_height` and `target_width`: Dimensions used to create the crop box for images.

## Training Process

1. **Data Preparation**:
   - Ensure your data is in the correct format and location as specified in the configuration file.
   - For 2D images, the pipeline will perform normalization. 3D images are not normalized.

2. **Model Selection**:
   - Specify the model in `selected_model_name`. The current example uses "Conv3DModel".

3. **Training Phases**:
   - The pipeline uses a nested cross-validation approach.
   - Training and validation phases are handled within `training_fold.py`.

4. **Performance Monitoring**:
   - The pipeline tracks accuracy and loss for each epoch.
   - Final accuracy and loss are reported at the end of training.

## Hyperparameter Tuning

Adjust hyperparameters in the configuration file to optimize model performance:

- Increase `batch_size` for faster training, but be mindful of memory constraints.
- Adjust `learning_rate` to control the step size of optimization.
- Modify `epochs` and `patience` to control training duration and early stopping.

## Preventing Overfitting

1. **Early Stopping**: 
   - The `patience` parameter in the configuration helps prevent overfitting by stopping training when performance plateaus.

2. **Learning Rate Scheduling**:
   - The pipeline implements a learning rate scheduler to adjust the learning rate during training.

## Supercomputer Considerations

1. **Memory Management**:
   - Be cautious with `batch_size` and model complexity to avoid exceeding available memory.
   - If the pipeline crashes, it may be due to insufficient memory. Try reducing `batch_size` or using a simpler model.

2. **Checkpointing**:
   - Use `k_epoch_checkpoint_frequency` to save model states regularly. This allows resuming training if interrupted.

3. **Resource Allocation**:
   - Ensure you request appropriate resources (CPU, GPU, memory) when submitting jobs to the supercomputer.

## Best Practices

1. **Reproducibility**: 
   - Set a fixed `seed` for reproducible results.

2. **Data Shuffling**: 
   - Use `shuffle_the_images` to randomize the order of training samples, which can improve model generalization.

3. **Monitoring**: 
   - Regularly check the training output for signs of overfitting or underfitting.

4. **Experimentation**: 
   - Start with default parameters and gradually adjust based on model performance.

5. **Version Control**: 
   - Keep track of different configuration files and their corresponding results for easy comparison and reproducibility.

6. **Model Selection**: 
   - Ensure that the `selected_model_name` in your configuration file matches one of the models defined in `src/model_processing/model_creator.py`.

7. **Memory Management**:
   - If you encounter "out of memory" errors, try reducing the `batch_size` in your configuration file.

8. **Image Channels**:
   - Set `channels` to 1 for grayscale images and 3 for RGB images. Ensure this matches your input data format.

9. **Metrics Tracking**:
   - By default, the pipeline tracks accuracy and loss. If you need to track recall as well, add it to the `metrics` list in your configuration file.

10. **Cross-validation**:
    - Use `shuffle_the_folds` parameter to control whether the folds in cross-validation should be shuffled.

11. **Image Cropping**:
    - Adjust `target_height` and `target_width` to control the dimensions of the crop box for your images. Ensure these values are appropriate for your dataset and model architecture.

Remember, the optimal configuration may vary depending on your specific dataset and classification task. Don't hesitate to experiment with different settings to achieve the best results.
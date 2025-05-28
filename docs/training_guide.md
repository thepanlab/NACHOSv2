# Configurations File

The training process is controlled by two YAML configuration files.

The first YMAL file configures the main properties:

```yml
# hpo (HyperParameter Optimization using Random Search)
# If use_hpo: false
# it will verify that the configuration file
# has single values

use_hpo: true
# file with hyperparameter configurations
configuration_filepath: "/home/pcallec/NACHOS/nachosv2/training/training/config_files/OCT_small/hp_single_configuration.yml"

number_channels: 1

# csv file with the metadata: image_path, label, fold
path_metadata_csv: "/home/pcallec/NACHOS/nachosv2/training/metadata_csv/pig_kidney_subset_metadata.csv"
# results output
output_path: "/home/pcallec/NACHOS/results/pig_kidney_subset"
# job_name is used as suffix for some files
job_name: "pig_kidney_subset"

checkpoint_epoch_frequency: 2

do_normalize_2d: true
do_shuffle_the_images: true
do_shuffle_the_folds: false

# If use_mixed_precision: true,
#  the model will be trained using mixed precision
# (fp16/fp32). This can speed up training and reduce
use_mixed_precision: false

class_names:
  - "cortex"
  - "medulla"
  - "pelvis-calyx"

# Accuracy is used by default, add more to analysis
metrics_list: []

fold_list:
  - "k1"
  - "k2"
  - "k3"
  - "k4"
  - "k5"

test_fold_list:
  - "k1"

# validation_fold_list is used only for 
# cross-validation
validation_fold_list:
  - "k1"
  - "k2"
  - "k3"

# if images are of different sizes,
# they will be resized to the target size

# if image is 2D. First dimension is height, and second is width
target_dimensions:
  - 301
  - 235

# Only useful in cross-validation, default is false
# If true, in addition to the predictions on the validation fold,
# the predictions on the test fold will be computed
enable_prediction_on_test: true
```

* `use_hpo`: `true` if hyperparameter optimization is used. If not, just provide single values inside `configuration_filepath`
* `configuration_filepath`: hyperparameter YAML file
* `number_channels`: number of channels for images. Grayscale: 1, RGB: 3.
* `path_metadata_csv`: csv file that would be used to extract fold id, label, and image filepath. An example how to get it can be found at [this link](../nachosv2/training/hpo/hpo_default_values.csv)
* `output_path`: output folder to place results
* `checkpoint_epoch_frequency`: checkpoint frequency saving
* `do_normalize_2d`: `true`` if data would be normalize with mean and standard deviation from training partition
* `do_shuffle_the_images`: `true` if data should be shuffled for each training epoch
* `use_mixed_precision`: `true` enables mixed precision (fp16/fp32) while training
* `class_names`: include class names in the order used for `path_metadata_csv`
* `metrics_list`: add metrics to be calculated during training
* `fold_list`: folds to be used for training, validation or test
* `test_fold_list`: folds to be used for test
* `validation_fold_list`: folds to be used for validation
*  `target_dimensions`: list of dimensions. If images are smaller or larger, they will be reshaped.
* `enable_prediction_on_test`: `true` if want to have results for test when doing cross-validation loop

The second YAML controls the hyperparameter configurations:

```yml
# only used when use_hpo: true
n_combinations: 9

# Specify single value or range e.g.
# if using a single value use
# batch_size: 32 OR
# batch_size: 
#   - 32 
# if using a range specify the min and max values
# batch_size:
#   - 16
#   - 128
batch_size:
  min: 16
  max: 128

do_cropping: false
# x and y cropping position
cropping_position:
  x: 40
  y: 10

# Specify single value or range
n_epochs: 5

# Specify single value or range
patience: 20

# Specify single value or range
learning_rate:
  min: 0.0001
  max: 0.01

# Specify scheduler learning rate
# If not specified learning rate is constant
learning_rate_scheduler: "InverseTimeDecay"
learning_rate_scheduler_parameters:
  decay: 0.01

# Specify single value or range
# if you don't want to use momentum, place 0 or omit value
# e.g. momentum: -1  
momentum:
  - 0.5
  - 0.9
  - 0.99

# Specify single value or range
enable_nesterov:
  - true
  - false

# Specify single value or range
architecture:
  - InceptionV3
  - ResNet50
```

* `n_combinations`: number of hyperparameters configurations to generate
* `batch_size`: provide single value, list or min and max values (power of two).
* `do_cropping`: true if cropping image, use `cropping_position`
* `cropping_position`: `x`: value, `y`: value
* `n_epochs`: provide single value, list or min and max values (multiple of 10)
* `patience`: provide single value, list or min and max values
* `learning_rate`: provide single value, list or min and max values (power of ten).
* `learning_rate_scheduler`: specified the learning rate scheduler
* `learning_rate_scheduler_parameters`: specified the learning rate scheduler parameters
* `momentum`: provide single value, list or min and max values
* `enable_nesterov`: value or list of values (true or false).
* `architecture`: single value or list of values for architecture
/home/pcallec/NACHOS/nachosv2/training/hpo/hpo_default_values.csv

If not specified, default values are retrieved from [this file](../nachosv2/training/hpo/hpo_default_values.csv)

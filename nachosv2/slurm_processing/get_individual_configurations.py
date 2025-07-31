from nachosv2.setup.command_line_parser import parse_command_line_args
from nachosv2.setup.get_config import get_config
from nachosv2.setup.utils import determine_if_cv_loop
from nachosv2.data_processing.check_unique_subjects import check_unique_subjects
from nachosv2.training.training.training import create_loop_indices
from nachosv2.setup.utils_processing import save_dict_to_yaml
from pathlib import Path


def create_individual_hyperparameter_configurations(config_dict,
                                                    hp_config):
    """
    Creates and saves an individual hyperparameter configuration as a YAML file.

    Args:
        config_dict (dict): Contains metadata like output folder and filename suffix.
        hp_config (dict): Dictionary of hyperparameter settings.

    Returns:
        dict: The created hyperparameter configuration dictionary.
    """

    index = hp_config["hp_config_index"]
    output_folder = Path(config_dict["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    pathfile = output_folder / f"hp_config_{index}.yaml"

    if pathfile.exists():
        print(f"Hyperparameter configuration file already exists {pathfile}, skipping creation.")
        return

    hp_dict = {}
    # Direct mapping of selected keys from hp_config to hp_dict
    hp_dict["n_combinations"] = 1
    keys_to_copy = [
        "batch_size", "do_cropping", "n_epochs", "learning_rate",
        "learning_rate_scheduler", "momentum", "enable_nesterov", "architecture"
    ]
    hp_dict = {key: hp_config[key] for key in keys_to_copy}
       
    # Handle scheduler parameters if present
    scheduler_params = hp_config.get("learning_rate_scheduler_parameters")
    if scheduler_params:
        hp_dict["learning_rate_scheduler_parameters"] = dict(scheduler_params)
        
    
    # Save as YAML file
    save_dict_to_yaml(hp_dict, pathfile)
    
    return hp_dict


def create_individual_training_configurations(config_dict,
                                              training_config_dict,
                                              item,
                                              index):
    """
    Creates and saves individual training configurations based on the provided
    configuration dictionary and hyperparameter settings.

    Args:
        config_dict (dict): Contains metadata like output folder and filename suffix.
        item (dict): Contains hyperparameter configuration and fold indices.

    Returns:
        dict: The created training configuration dictionary.
    """

    individual_training_dict = dict()
    
    individual_training_dict["use_hpo"] = False

    hp_index = item["hp_configuration"]["hp_config_index"]
    output_folder = Path(config_dict["output_folder"])
    pathfile = output_folder / f"hp_config_{hp_index}.yaml"

    individual_training_dict["configuration_filepath"] = str(pathfile)
    
    keys_to_copy = [
        "number_channels", "path_metadata_csv", "output_path", "job_name",
        "checkpoint_epoch_frequency", "do_normalize_2d", "do_shuffle_the_images", "use_mixed_precision", "class_names", "metrics_list", "fold_list", "target_dimensions", "enable_prediction_on_test"
    ]

    individual_training_dict.update({key: training_config_dict[key] for key in keys_to_copy})
    
    individual_training_dict["test_fold_list"] = item["test"]
    individual_training_dict["validation_fold_list"] = item["validation"]
    
    output_folder = Path(config_dict["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    suffix = config_dict["suffix_filename"]
    pathfile = output_folder / f"{suffix}_{index}.yaml"

    save_dict_to_yaml(individual_training_dict, pathfile)
    
    return individual_training_dict


def main():
    
    args = parse_command_line_args()
    
    loop = args["loop"]
    training_config_dict = get_config(args['file'])
    config_dict = get_config(args['config_individual'])
    
    is_cv_loop = determine_if_cv_loop(loop)

    if is_cv_loop:  # Only if we are in the inner loop
        check_unique_subjects(training_config_dict["validation_fold_list"],
                              "validation")

    # Create the list of fold combinations for training
    indices_loop_list = create_loop_indices(training_config_dict,
                                            is_cv_loop)
    
    hp_set = set()
    
    for index, item in enumerate(indices_loop_list):
        
        # create hyperparameter configuration
        hp_config = item["hp_configuration"]
        hp_index = hp_config["hp_config_index"]
        if hp_index not in hp_set:
            create_individual_hyperparameter_configurations(config_dict,
                                                            hp_config)
        hp_set.add(hp_index)
        
        # create training configuration
        create_individual_training_configurations(config_dict,
                                                  training_config_dict,
                                                  item,
                                                  index)
        
        
if __name__ == "__main__":
    main()
    
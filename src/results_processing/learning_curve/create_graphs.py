from matplotlib import pyplot as plt
import os
import pandas as pd
import regex as re
from termcolor import colored

from src.results_processing.learning_curve.learning_curve_utils import get_subject_name


def create_graphs(file_list, file_path, results_path, results_config):
    """
    Creates graphs for each item

    Args:
        file_list (list): A list of all files to process.
        file_path (str): File output path.
        results_path (ste): Path to output the file to.
        results_config (dict): A configuration.
    """
    
    # Looping through each file and creating needed graphs
    for file in file_list:
        
        # Puts the CSV file into a dataframe
        results_dataframe = pd.read_csv(os.path.join(file_path, file), index_col = 0)
        file_name = re.sub('.csv', '', file)

        
        # Gets the current subject
        subject_name = get_subject_name(file_name)
        
        
        # Establishing the plot's font and font sizes
        font_family = results_config['font_family']
        label_font_size = results_config['label_font_size']
        title_font_size = results_config['title_font_size']
        
        # Establishing the plot's saving format and resolution
        save_format = results_config['save_format']
        save_res = results_config['save_resolution']
        
        
        # Loops throught the metrics to create the graphs
        for metric_name in ['accuracy', 'loss']:
            
            # Grabs the training and validation metric data from the dataframe
            train_metric = results_dataframe[f'train_{metric_name}']
            try:
                validation_metric = results_dataframe[f'validation_{metric_name}']
                
            except:
                validation_metric = results_dataframe[f'train_{metric_name}']
            
            
            # Sets the font parameters
            plt.rcParams['font.family'] = font_family
            plt.rc('axes', labelsize = label_font_size)
            plt.rc('axes', titlesize = title_font_size)
            
            # Establises the plot's colors
            training_line_color = results_config[f'training_{metric_name}_line_color']
            validation_line_color = results_config[f'validation_{metric_name}_line_color']
            
            
            # Creates the plot
            plt.plot(train_metric, color = training_line_color)                 # The training curve
            plt.plot(validation_metric, color = validation_line_color)          # The validation curve
            plt.title(f'{subject_name.upper()} {metric_name} learning curve')   # The title
            plt.ylabel(metric_name)                                             # The y label (metric)
            plt.xlabel('epoch')                                                 # The x label (epoch)
            plt.legend(['training', 'validation'], loc = 'best')                # The legend
            
            
            # Removes top and right axis lines
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.right'] = False


            # Saves the figure
            plt.savefig(os.path.join(results_path, f"{file_name}_lc_{metric_name}.{save_format}"), format = save_format, dpi = save_res)
            plt.close()
            
            print(colored(f"{metric_name} learning curve has been created for: {file_name}", 'green'))


        print(colored("\n"), end = "")

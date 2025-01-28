from pathlib import Path

import pandas as pd    
    
def get_class_names_from_prediction_path(config: dict,
                                         path: Path) -> pd.DataFrame:
    
    # take path prediction and get class name file
    class_name_filepath = Path(str(path).replace('prediction_results.csv',
                                                 'class_names.csv'))
    
    df_class_names = pd.read_csv(class_name_filepath,
                                 index_col=0)
    
    return df_class_names
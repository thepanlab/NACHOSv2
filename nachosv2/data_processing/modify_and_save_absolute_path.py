import argparse
from pathlib import Path
import pandas as pd
from functools import partial

def modify_absolute_path(absolute_path:str,
                         selected_folder:str,
                         new_parent_folder: Path):
    
    absolute_path = Path(absolute_path)
    for i, part in enumerate(absolute_path.parts):
        if part == selected_folder:
            new_path = new_parent_folder / Path(*absolute_path.parts[i:])
            return str(new_path)


def modify_metadata(csv_path: Path,
                    selected_folder: str,
                    new_parent_folder: Path):
    # Load the CSV
    df = pd.read_csv(csv_path)
    # Modify the absolute_filepath column
    modify_fn = partial(modify_absolute_path, selected_folder=selected_folder, new_parent_folder=new_parent_folder)
    df['absolute_filepath'] = df['absolute_filepath'].apply(modify_fn) 

    # Save the updated CSV
    cvs_new_path = new_parent_folder / csv_path.name
    df.to_csv(cvs_new_path, index=False)
    print(f"New metadata CSV saved at: {cvs_new_path}")
    return str(cvs_new_path)


def main():
    parser = argparse.ArgumentParser()

    # Definition of all arguments
    parser.add_argument( # Allows to specify the config file in command line
        '--csv_path',
        type = str, default = None, required = True,
        help = 'Path for metadata CSV file.'
    )

    parser.add_argument( # Allows to specify the config folder in command line
        '--selected_folder',
        type = str, default = None, required = True,
        help = 'Folder in the absolute path to keep'
    )

    parser.add_argument( # Allows to specify the config folder in command line
        '--new_parent_folder',
        type = str, default = None, required = True,
        help = 'Path to add before selected folder'
    )

    args = parser.parse_args()

    return modify_metadata(csv_path=Path(args.csv_path),
    selected_folder=args.selected_folder,
    new_parent_folder=Path(args.new_parent_folder))


if __name__ == "__main__":
    main()
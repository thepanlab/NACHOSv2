import argparse
from pathlib import Path
import pandas as pd
from functools import partial


def update_path_starting_from_folder(
    absolute_path:str,
    retain_subpath_from:str,
    updated_parent_dir: Path):
    """
    Reconstruct the filepath by replacing the part before the specified folder
    `retain_subpath_from` with a new parent directory path `updated_parent_dir`.

    Args:
        absolute_path (str): Original absolute file path.
        folder_to_retain (str): Name of the folder from which to retain the path.
        updated_parent_dir (Path): New parent directory to prepend.

    Returns:
        str: Modified file path with updated parent directory.
    """
    
    absolute_path = Path(absolute_path)
    for i, part in enumerate(absolute_path.parts):
        if part == retain_subpath_from:
            new_path = updated_parent_dir / Path(*absolute_path.parts[i:])
            return str(new_path)


def update_metadata_csv(csv_filepath: Path,
                        retain_subpath_from: str,
                        updated_parent_dir: Path):
    """
    Update the absolute filepaths in a metadata CSV by modifying the parent directories.

    Args:
        csv_file_path (Path): Path to the CSV metadata file.
        retain_subpath_from (str): Folder name from which the remaining path should be preserved.
        updated_parent_dir (Path): New parent path to prepend before `retain_subpath_from`.

    Returns:
        str: Path to the updated metadata CSV file.
    """
    # Load the CSV
    # Load metadata CSV into a DataFrame
    df = pd.read_csv(csv_filepath)

    # Partially apply the path update function
    update_fn  = partial(update_path_starting_from_folder,
                        retain_subpath_from=retain_subpath_from,
                        updated_parent_dir=updated_parent_dir)
    df['absolute_filepath'] = df['absolute_filepath'].apply(update_fn ) 

     # Save updated DataFrame to a new CSV file in the new parent directory path
    cvs_new_path = updated_parent_dir / csv_filepath.name
    df.to_csv(cvs_new_path, index=False)
    print(f"New metadata CSV saved at: {cvs_new_path}")
    return str(cvs_new_path)


def main():
    parser = argparse.ArgumentParser()

    # Definition of all arguments
    parser.add_argument( # Allows to specify the config file in command line
        '--csv_filepath',
        type = str, default = None, required = True,
        help = 'Path to the metadata CSV file.'
    )

    parser.add_argument( # Allows to specify the config folder in command line
        '--retain_subpath_from',
        type = str, default = None, required = True,
        help = 'Folder in the path from which to retain the rest of the structure.'
    )

    parser.add_argument( # Allows to specify the config folder in command line
        '--updated_parent_dir',
        type = str, default = None, required = True,
        help = 'New parent directory to prepend before the retained subpath.'
    )

    args = parser.parse_args()

    return update_metadata_csv(
        csv_filepath=Path(args.csv_filepath),
        retain_subpath_from=args.retain_subpath_from,
        updated_parent_dir=Path(args.updated_parent_dir)
        )


if __name__ == "__main__":
    main()
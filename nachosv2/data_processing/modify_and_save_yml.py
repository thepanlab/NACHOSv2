import argparse
from pathlib import Path
import yaml

def modify_yaml_value(yaml_path: Path,
                      key_to_modify: str,
                      new_value: str,
                      output_folder: Path = None):

    # Read the YAML file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Modify the value if the key exists
    if key_to_modify in data:
        data[key_to_modify] = new_value
    else:
        print(f"Key '{key_to_modify}' not found in the YAML file.")
        return

    # Write back to the same file (or change this to a new path)
    if output_folder:
        yaml_path = output_folder / yaml_path.name
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Updated '{key_to_modify}' to: {new_value}")


def main():
    
    parser = argparse.ArgumentParser()

    # Definition of all arguments
    parser.add_argument( # Allows to specify the config file in command line
        '--yml_path',
        type = str, default = None, required = True,
        help = 'Path for yml file.'
    )
    
    parser.add_argument( # Allows to specify the config file in command line
        '--output_folder',
        type = str, default = None, required = False,
        help = 'Output folder path for new yml file.'
    )
    
    parser.add_argument( # Allows to specify the config file in command line
        '--key',
        type = str, default = None, required = True,
        help = 'Output folder path for new yml file.'
    )

    parser.add_argument( # Allows to specify the config file in command line
        '--value',
        type = str, default = None, required = True,
        help = 'Output folder path for new yml file.'
    )

    args = parser.parse_args()
    
    modify_yaml_value(yaml_path= Path(args.yml_path),
                      key_to_modify=args.key,
                      new_value=args.value,
                      output_folder=Path(args.output_folder) if args.output_folder else None)


if __name__ == "__main__":
    main()
    
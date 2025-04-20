import argparse
from pathlib import Path
import yaml


def update_yaml_key_value(yaml_path: Path,
                          target_key: str,
                          updated_value: str,
                          output_dir: Path = None):

    # Read the YAML file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Modify the value if the key exists
    if target_key in data:
        data[target_key] = updated_value
    else:
        print(f"Key '{target_key}' not found in the YAML file.")
        return

    # Write back to the same file (or change this to a new path)
    if output_dir:
        yaml_path = output_dir / yaml_path.name
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Updated '{target_key}' to: {updated_value}. Save at {yaml_path}")


def main():
    
    parser = argparse.ArgumentParser()

    # Definition of all arguments
    parser.add_argument( # Allows to specify the config file in command line
        '--yaml_path',
        type = str, default = None, required = True,
        help = 'Path to the input YAML file.'
    )
    
    parser.add_argument( # Allows to specify the config file in command line
        '--output_dir',
        type = str, default = None, required = False,
        help = 'Output folder path for new yml file.'
               'If not provided, the original file will be overwritten.'
    )
    
    parser.add_argument( # Allows to specify the config file in command line
        '--key',
        type = str, default = None, required = True,
        help = 'Key in the YAML file whose value should be updated.'
    )

    parser.add_argument( # Allows to specify the config file in command line
        '--value',
        type = str, default = None, required = True,
        help = 'New value to assign to the specified key.'
    )

    args = parser.parse_args()
    
    update_yaml_key_value(
        yaml_path= Path(args.yaml_path),
        target_key=args.key,
        updated_value=args.value,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )


if __name__ == "__main__":
    main()

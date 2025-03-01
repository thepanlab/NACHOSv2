# file_checks.py
from pathlib import Path
from termcolor import colored

def ensure_path_exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(colored(f"Path {path} does not exist.", "red"))
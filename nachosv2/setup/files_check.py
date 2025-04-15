# file_checks.py
from pathlib import Path
from termcolor import colored


def ensure_path_exists(path: str|Path):
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(colored(f"Path {path} does not exist.", "red"))
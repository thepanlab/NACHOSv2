import os
import sys


"""
This script sets up the project paths by adding relevant directories to sys.path.
This enables easy importing of modules from these directories throughout the project.
"""

# Gets the absolute path of the directory containing setup_paths.py
project_root = os.path.dirname(os.path.abspath(__file__))

# Adds the project root directory to sys.path if it's not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
    
# Adds the 'scripts' directory to sys.path
scripts_path = os.path.join(project_root, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Adds the 'src' directory to sys.path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

#!/bin/bash
#
# Usage:
#       setup_env.sh [-d]
#
# Install the package together with its dependencies as well as the `dev` dependencies 
# (see [project.optional-dependencies])
#
# [-d] Force removal of the created venv.



set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Name of the virtual environment folder
VENV_DIR="$SCRIPT_DIR/venv"

# 
if [ "$1" == "-d" ];
then
    rm -rf "$VENV_DIR"
fi

which python3
python3 --version

# Check if virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
else
    # Create a virtual environment
    python3 -m venv $VENV_DIR
    echo "Virtual environment created."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

Install dependencies from requirements.txt if the file exists
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Dependencies installed from requirements.txt."
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

# Install your package in editable mode with development extras
pip install -e .
echo "Package installed in editable mode with development dependencies."


# Extract the project name from pyproject.toml using Python
# PROJECT_NAME=$(python3 -c "import sys, importlib.util; \
# tomllib = __import__('tomllib') if importlib.util.find_spec('tomllib') is not None else __import__('toml'); \
# f = open('pyproject.toml', 'rb'); config = tomllib.load(f); f.close(); \
# sys.stdout.write(config['project']['name'])")
# 
# 
# # Register the Jupyter kernel using the project name
# python3 -m ipykernel install --user --name="${PROJECT_NAME}_env" --display-name "${PROJECT_NAME} Environment"
# echo "Jupyter kernel installed as ${PROJECT_NAME}_env."

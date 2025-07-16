#!/bin/bash

# Script to activate the conda environment for the mlTrainer project

echo "Activating conda environment..."
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate base

echo "Python environment activated!"
echo "Python version: $(python --version)"
echo "Available packages:"
echo "- numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "- pandas: $(python -c 'import pandas; print(pandas.__version__)')"
echo "- scikit-learn: $(python -c 'import sklearn; print(sklearn.__version__)')"
echo "- requests: $(python -c 'import requests; print(requests.__version__)')"
echo "- anthropic: $(python -c 'import anthropic; print(anthropic.__version__)')"

echo ""
echo "To use this environment in your shell, run:"
echo "source activate_env.sh"
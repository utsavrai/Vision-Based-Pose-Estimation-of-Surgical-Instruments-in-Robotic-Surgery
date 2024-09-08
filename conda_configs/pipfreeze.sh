#!/bin/bash

# Get the list of all conda environments
environments=$(conda info --envs | awk '{print $1}' | grep -v '^#' | grep -v 'base')

# Loop through each environment
for env in $environments
do
    echo "Processing environment: $env"
    
    # Activate the conda environment
    conda activate $env
    
    # Run pip freeze and save to a requirements.txt file in the current directory
    pip freeze > "${env}_requirements.txt"
    
    # Deactivate the conda environment
    conda deactivate
done

echo "All environments have been processed and requirements.txt files created."

#!/bin/bash

# List all environments
environments=$(conda info --envs | awk '{print $1}' | grep -v '^#' | grep -v 'base')

# Loop through each environment and export it to a YAML file
for env in $environments
do
    echo "Exporting environment: $env"
    conda env export -n $env > "${env}.yml"
done

echo "All environments have been exported."

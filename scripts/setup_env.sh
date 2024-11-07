#!/bin/bash
set -e

# Start timer
start_time=$(date +%s)

# Create environment name with the current date
env_prefix=ai_experimentation_env

# Create the conda environment


# Get the base directory of Conda
CONDA_ROOT=$(conda info --base)

# Check if conda.sh exists
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    # Use dot (.) instead of source for better compatibility
    . "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "Conda profile script not found at $CONDA_ROOT/etc/profile.d/conda.sh"
    exit 1
fi

current_shell=bash

# Map the output of 'ps' to the shell configuration file
case "$current_shell" in
    bash)
        shell_rc="$HOME/.bashrc"
        ;;
    zsh)
        shell_rc="$HOME/.zshrc"
        ;;
    sh)
        shell_rc="$HOME/.profile"
        ;;
    *)
        echo "Unknown shell: $current_shell. Please initialize manually."
        exit 1
        ;;
esac

# Check if conda is initialized for the current shell
if ! grep -q "conda initialize" "$shell_rc"; then
    echo "Conda is not initialized for $current_shell. Initializing now..."

    # Run conda init for the detected shell
    conda init $current_shell

    # Source the updated shell configuration file
    echo "Sourcing $shell_rc to apply changes..."
    . "$shell_rc"
else
    echo "Conda is already initialized for $current_shell."
fi

# Now, conda commands like `conda activate` should work
echo "Conda is ready to use in $current_shell."




if conda info --envs | grep -q "$env_prefix"; then
    echo "Environment '$env_prefix' already exists. Deleting..."
    conda env remove -n ai_experimentation_env
fi

echo "Creating environment '$env_prefix'..."
conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix
conda install git -y

# Install packages
pip install ninja
pip install --requirement requirements.txt



echo "Currently in env $(which python)"

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"

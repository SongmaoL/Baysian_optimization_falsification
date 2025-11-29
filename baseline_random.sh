#!/bin/bash

#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00

module purge
module load conda
module load apptainer

# 1. Ensure we are in the root of your repository
# Adjust this path to where your 'Baysian_optimization_falsification' folder is on the cluster
cd /path/to/your/Baysian_optimization_falsification

# 2. Run everything INSIDE the container
# We map the current directory ($PWD) to /app inside the container for easy access
# We use 'bash -c' to run multiple commands in sequence inside the container

singularity exec --nv \
  -B $PWD:/app \
  -B /scratch1 \
  /project/jdeshmuk_786/carla-0.9.15_4.11.sif \
  bash -c "
    # Setup environment inside container
    # Note: You might not need 'conda activate carla' if the container already has dependencies
    # or if you installed them in a local venv mapped into the container.
    # Assuming standard python setup:
    
    echo 'Starting CARLA server...'
    # The container usually has CARLA at /home/carla/ or /opt/carla-simulator/
    # Check the container docs, but based on your error, /home/carla/CarlaUE4.sh is likely correct inside the container
    /home/carla/CarlaUE4.sh -nosound -vulkan -RenderOffScreen &
    CARLA_PID=\$!
    
    sleep 20
    
    echo 'Running Falsification...'
    cd /app
    
    # Explicitly point to the project directory using absolute or relative path
    # Since we mapped $PWD to /app, and we are in the repo root:
    python project_example_code/falsification_framework.py \
        --strategy random_search \
        --n-iterations 200 \
        --output-dir falsification_output_random_200 \
        --checkpoint-interval 25 \
        --carla-project project_example_code/csci513-miniproject1
        
    # Cleanup
    kill \$CARLA_PID
"
